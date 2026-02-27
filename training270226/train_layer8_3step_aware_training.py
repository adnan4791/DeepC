#!/usr/bin/env python3
"""
train_layer8_3step_aware_training.py  (v4 — Race-Aware Compensation)

Phase 2 fine-tuning: starts from V3 trained weights and applies
Race-Aware gradient compensation.

Strategy:
  Step 1: Deconvolution — STE (C forward + PyTorch backward), same as V3
  Step 2: Imadd — RaceAwareStep:
    Forward:  safe sum (deterministic, stable)
    Backward: gradient amplified by 1/ratio where ratio = avg_race / safe_sum
              This forces the optimizer to strengthen weights where race
              conditions cause data loss.
  Step 3: Bias — standard PyTorch

The key insight:
  If pixel (x,y) loses 50% of its value due to race conditions (ratio=0.5),
  we amplify the gradient by 2× at that pixel. This pushes the optimizer
  to make the weights 2× stronger there, so even after 50% loss from
  race conditions, the output is still correct.
"""

import os
import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1. LOAD C LIBRARY (per-step functions)
# ============================================================================
lib = ctypes.CDLL(os.path.abspath("./fsrcnn_layer8_3step.so"))

# --- Weight/Bias accessors ---
lib.set_weights_layer8.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.set_weights_layer8.restype = None
lib.get_weights_layer8.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.get_weights_layer8.restype = None
lib.set_bias_layer8.argtypes = [ctypes.c_double]
lib.set_bias_layer8.restype = None
lib.get_bias_layer8.argtypes = []
lib.get_bias_layer8.restype = ctypes.c_double

# --- Step 1: Deconv all channels ---
lib.deconv_all_channels.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.deconv_all_channels.restype = None

# --- Step 2: Imadd with race condition ---
lib.imadd_race_all_channels.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.imadd_race_all_channels.restype = None

# --- Step 2b: Imadd safe ---
lib.imadd_safe_all_channels.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.imadd_safe_all_channels.restype = None

# --- Thread control ---
lib.set_num_threads.argtypes = [ctypes.c_int]
lib.set_num_threads.restype = None
lib.get_max_threads.argtypes = []
lib.get_max_threads.restype = ctypes.c_int

lib.set_num_threads(8)
print(f"OpenMP threads: {lib.get_max_threads()}")


# ============================================================================
# 2. HELPERS — C library wrappers
# ============================================================================
def c_deconv_forward(input_np, weights_np, num_ch, rows, cols, scale):
    """Call C deconv_all_channels for one sample. Returns (56, out_rows, out_cols)."""
    out_rows = rows * scale
    out_cols = cols * scale

    w_flat = weights_np.flatten().astype(np.float64)
    lib.set_weights_layer8(w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    in_flat = input_np.flatten().astype(np.float64)
    in_ptr = in_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_flat = np.zeros(num_ch * out_rows * out_cols, dtype=np.float64)
    out_ptr = out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.deconv_all_channels(in_ptr, out_ptr, rows, cols, scale)
    return out_flat.reshape(num_ch, out_rows, out_cols)


def call_c_imadd_race(input_tensor):
    """Run C imadd_race for a batch. Returns (B, 1, H, W) tensor."""
    B, C, H, W = input_tensor.shape
    out = torch.zeros((B, 1, H, W), dtype=torch.float64)
    for b in range(B):
        in_np = input_tensor[b].detach().cpu().numpy().flatten().astype(np.float64)
        out_np = np.zeros(H * W, dtype=np.float64)
        lib.imadd_race_all_channels(
            in_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            H, W, C
        )
        out[b, 0] = torch.from_numpy(out_np.reshape(H, W))
    return out


# ============================================================================
# 3. RACE-AWARE COMPENSATION AUTOGRAD FUNCTION
# ============================================================================
class RaceAwareStep(torch.autograd.Function):
    """
    Race-Aware Compensation for Step 2 (imadd).

    Forward: Uses safe sum (deterministic) → stable training, no random loss jumps
    Backward: Gradient is amplified by 1/ratio where:
      ratio = average_race_output / safe_sum_output
      - ratio ≈ 1.0 → no race impact → normal gradient
      - ratio < 1.0 → data lost by race → gradient amplified to compensate
      - ratio > 1.0 → race added extra → gradient dampened

    The race condition is sampled N_SAMPLES times and averaged to get a
    statistical estimate of the survival rate at each pixel.
    """
    N_SAMPLES = 3  # Number of race condition samples to average

    @staticmethod
    def forward(ctx, x):
        # 1. Deterministic safe sum (this is the forward output)
        safe_sum = x.sum(dim=1, keepdim=True)

        # 2. Sample race condition multiple times for statistical estimate
        with torch.no_grad():
            race_samples = []
            for _ in range(RaceAwareStep.N_SAMPLES):
                race_samples.append(call_c_imadd_race(x))
            avg_race = torch.stack(race_samples).mean(dim=0)

            # 3. Compute survival ratio: race_output / safe_output
            #    ratio = 1.0 means no race impact
            #    ratio < 1.0 means data is lost
            ratio = avg_race / (safe_sum.detach() + 1e-8)
            ratio = torch.clamp(ratio, 0.1, 1.5)  # Prevent extreme values

        ctx.save_for_backward(ratio)
        return safe_sum  # Stable, deterministic output

    @staticmethod
    def backward(ctx, grad_output):
        ratio, = ctx.saved_tensors
        # Compensation: amplify gradient where data is lost (ratio < 1)
        # If ratio = 0.5, gradient becomes 2× → optimizer pushes weights higher
        # Add small eps to prevent division by zero
        compensated_grad = grad_output / (ratio + 1e-2)
        # Expand from (B, 1, H, W) back to (B, 56, H, W)
        return compensated_grad.expand(-1, 56, -1, -1)


# ============================================================================
# 4. MODEL — Race-Aware with STE Deconv
# ============================================================================
class Layer8FSRCNN_RaceAware(nn.Module):
    """
    Layer 8 model with Race-Aware compensation:
      Step 1: Deconvolution via STE (C forward, PyTorch backward)
      Step 2: RaceAwareStep (safe sum forward, compensated backward)
      Step 3: Bias addition
    """

    def __init__(self, scale, num_channels=56, kernel_size=9):
        super().__init__()
        self.scale = scale
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.deconv_weights = nn.Parameter(
            torch.zeros(num_channels, 1, kernel_size, kernel_size, dtype=torch.float64)
        )
        self.bias = nn.Parameter(torch.tensor([-0.03262640000], dtype=torch.float64))

        # Store starting weights for regularization
        self.register_buffer('start_weights', None)

        self._load_weights()

    def _load_weights(self):
        # Try to load V3 trained weights first (fine-tuning mode)
        if os.path.exists("weights_layer8_trained.txt"):
            w = np.loadtxt("weights_layer8_trained.txt", dtype=np.float64)
            w = w.reshape(self.num_channels, 1, self.kernel_size, self.kernel_size)
            self.deconv_weights.data = torch.from_numpy(w)
            self.start_weights = torch.from_numpy(w.copy())
            print(f"✓ Loaded weights_layer8_trained.txt (V3 fine-tuning mode, {len(w.flatten())} weights)")
        elif os.path.exists("weights_layer8_original.txt"):
            w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64)
            w = w.reshape(self.num_channels, 1, self.kernel_size, self.kernel_size)
            self.deconv_weights.data = torch.from_numpy(w)
            self.start_weights = torch.from_numpy(w.copy())
            print(f"✓ Loaded weights_layer8_original.txt (from scratch, {len(w.flatten())} weights)")
        else:
            print("⚠ No weight file found, using zero init")

        # Load bias if available
        if os.path.exists("biasess_layer8_trained.txt"):
            b = float(open("biasess_layer8_trained.txt").read().strip())
            self.bias.data = torch.tensor([b], dtype=torch.float64)
            print(f"✓ Loaded bias: {b:.10f}")

    def weight_regularization_loss(self):
        """L2 distance from starting weights."""
        if self.start_weights is not None:
            return ((self.deconv_weights - self.start_weights) ** 2).mean()
        return torch.tensor(0.0, dtype=torch.float64)

    def forward(self, x):
        """
        x: (B, 56, rows, cols)
        Returns: (B, 1, out_rows, out_cols)
        """
        batch_size, num_ch, rows, cols = x.shape
        out_rows = rows * self.scale
        out_cols = cols * self.scale
        padding = self.kernel_size // 2
        output_padding = self.scale - 1

        # =================================================================
        # STEP 1: Deconvolution — STE (C forward, PyTorch backward)
        # =================================================================
        pt_deconv = torch.nn.functional.conv_transpose2d(
            x, self.deconv_weights,
            bias=None, stride=self.scale, padding=padding,
            output_padding=output_padding, groups=num_ch
        )
        pt_deconv = pt_deconv[:, :, :out_rows, :out_cols]

        with torch.no_grad():
            w_np = self.deconv_weights.detach().cpu().numpy()
            c_deconv = torch.zeros_like(pt_deconv)
            for b in range(batch_size):
                in_np = x[b].detach().cpu().numpy()
                out_np = c_deconv_forward(in_np, w_np, num_ch, rows, cols, self.scale)
                c_deconv[b] = torch.from_numpy(out_np)

        deconv_out = pt_deconv + (c_deconv - pt_deconv).detach()

        # =================================================================
        # STEP 2: Imadd — Race-Aware Compensation
        # =================================================================
        imadd_out = RaceAwareStep.apply(deconv_out)

        # =================================================================
        # STEP 3: Bias
        # =================================================================
        output = imadd_out + self.bias

        return output


# ============================================================================
# 5. DATASET
# ============================================================================
class FSRCNNDataset(Dataset):
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "dimensions.txt"), "r") as f:
            self.in_rows, self.in_cols, self.scale, self.num_frames = map(int, f.read().split())
        self.out_rows = self.in_rows * self.scale
        self.out_cols = self.in_cols * self.scale
        print(f"✓ Dataset: {self.num_frames} frames, "
              f"input={self.in_rows}x{self.in_cols}, "
              f"output={self.out_rows}x{self.out_cols}, scale={self.scale}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        l7 = np.fromfile(
            os.path.join(self.data_dir, f"layer7_frame{idx:04d}.bin"),
            dtype=np.float64
        ).reshape(56, self.in_rows, self.in_cols)

        hr = np.fromfile(
            os.path.join(self.data_dir, f"hr_frame{idx:04d}.bin"),
            dtype=np.float64
        ).reshape(1, self.out_rows, self.out_cols)

        return torch.from_numpy(l7), torch.from_numpy(hr)


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def compute_psnr(mse_loss):
    if mse_loss <= 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse_loss)


def train():
    print("=" * 65)
    print("  FSRCNN Layer 8 — Race-Aware Compensation Training (V4)")
    print("=" * 65)

    dataset = FSRCNNDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = Layer8FSRCNN_RaceAware(scale=dataset.scale)

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    num_epochs = 100  # Fewer epochs — this is fine-tuning
    reg_lambda = 5e-3  # Stronger regularization since we start from good weights

    # Fine-tuning LR — much smaller than V3
    optimizer = optim.Adam([
        {'params': [model.deconv_weights], 'lr': 1e-5},
        {'params': [model.bias], 'lr': 1e-5},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_loss = float('inf')
    patience = 0
    max_patience = 30

    print(f"\nConfig: epochs={num_epochs}, batch=8, lr=1e-5 (fine-tuning)")
    print(f"        grad_clip=0.3, loss=0.7*MSE + 0.3*L1, reg_lambda={reg_lambda}")
    print(f"        Step2=RaceAware (safe_sum fwd + compensated bwd, {RaceAwareStep.N_SAMPLES} samples)")
    print("-" * 65)

    for epoch in range(num_epochs):
        model.train()
        epoch_mse = 0.0
        epoch_l1 = 0.0
        num_batches = 0

        for batch_idx, (layer7_input, hr_target) in enumerate(dataloader):
            optimizer.zero_grad()

            output = model(layer7_input)

            loss_mse = mse_criterion(output, hr_target)
            loss_l1 = l1_criterion(output, hr_target)
            loss_reg = model.weight_regularization_loss()
            loss = 0.7 * loss_mse + 0.3 * loss_l1 + reg_lambda * loss_reg

            loss.backward()

            # Tighter gradient clipping for fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

            optimizer.step()

            with torch.no_grad():
                model.bias.data.clamp_(-0.5, 0.5)

            epoch_mse += loss_mse.item()
            epoch_l1 += loss_l1.item()
            num_batches += 1

        scheduler.step()

        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_loss = 0.7 * avg_mse + 0.3 * avg_l1
        current_lr = scheduler.get_last_lr()[0]
        psnr_est = compute_psnr(avg_mse)
        w_dist = model.weight_regularization_loss().item()

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"MSE: {avg_mse:.8f} | L1: {avg_l1:.6f} | "
              f"PSNR~{psnr_est:.2f}dB | LR: {current_lr:.2e} | "
              f"Bias: {model.bias.item():.8f} | Wdist: {w_dist:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            save_weights(model)
            print(f"  → ★ Best model saved! (loss={best_loss:.8f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {max_patience} epochs)")
                break

    print("\n" + "=" * 65)
    print(f"  Training complete! Best combined loss: {best_loss:.8f}")
    print(f"  Weights saved to: weights_layer8_race_aware.txt")
    print(f"  Bias saved to:    biasess_layer8_race_aware.txt")
    print("=" * 65)


def save_weights(model):
    weights = model.deconv_weights.data.numpy().flatten()
    bias = model.bias.data.item()

    with open("weights_layer8_race_aware.txt", "w") as f:
        np.savetxt(f, weights[None, :], fmt="  %.16e", delimiter="  ")

    with open("biasess_layer8_race_aware.txt", "w") as f:
        f.write(f"{bias:.10f}\n")


if __name__ == "__main__":
    train()
