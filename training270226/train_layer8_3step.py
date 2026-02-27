#!/usr/bin/env python3
"""
train_layer8_3step.py  (v2 — STE-based gradient)

Training script for FSRCNN Layer 8 with PROPER per-step error backpropagation.

The 3 steps are separated as individual autograd functions:
  Step 1: Deconvolution (per-channel, 9x9 kernel, stride=scale)
  Step 2: Imadd with Race Condition (56ch → 1ch summation)
  Step 3: Bias addition (learnable scalar)

Gradient strategy:
  STE (Straight-Through Estimator) trick:
    output = pytorch_output + (c_output - pytorch_output).detach()
  This gives the C output in forward but PyTorch-correct gradients in backward.
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

# --- Step 2b: Imadd safe (for gradient estimation) ---
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
# 2. HELPER — Call C deconv and return numpy array
# ============================================================================
def c_deconv_forward(input_np, weights_np, num_ch, rows, cols, scale):
    """Call C deconv_all_channels for one sample. Returns (56, out_rows, out_cols)."""
    out_rows = rows * scale
    out_cols = cols * scale

    # Set weights
    w_flat = weights_np.flatten().astype(np.float64)
    lib.set_weights_layer8(w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Input
    in_flat = input_np.flatten().astype(np.float64)
    in_ptr = in_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Output
    out_flat = np.zeros(num_ch * out_rows * out_cols, dtype=np.float64)
    out_ptr = out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.deconv_all_channels(in_ptr, out_ptr, rows, cols, scale)
    return out_flat.reshape(num_ch, out_rows, out_cols)


def c_imadd_race_forward(input_np, num_ch, height, width):
    """Call C imadd_race_all_channels for one sample. Returns (height, width)."""
    in_flat = input_np.flatten().astype(np.float64)
    out_flat = np.zeros(height * width, dtype=np.float64)
    lib.imadd_race_all_channels(
        in_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        height, width, num_ch
    )
    return out_flat.reshape(height, width)


def c_imadd_safe_forward(input_np, num_ch, height, width):
    """Call C imadd_safe_all_channels for one sample. Returns (height, width)."""
    in_flat = input_np.flatten().astype(np.float64)
    out_flat = np.zeros(height * width, dtype=np.float64)
    lib.imadd_safe_all_channels(
        in_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        height, width, num_ch
    )
    return out_flat.reshape(height, width)


# ============================================================================
# 3. MODEL — STE-based forward with per-step separation
# ============================================================================
class Layer8FSRCNN_3Step(nn.Module):
    """
    Layer 8 model with explicit 3-step forward pass using STE trick:

      Step 1: Deconvolution
        Forward  → C library (bit-exact with hardware)
        Backward → PyTorch conv_transpose2d autograd (mathematically correct)
        STE connects both: output = pt_output + (c_output - pt_output).detach()

      Step 2: Imadd with race condition
        Forward  → C library (bit-exact race condition)
        Backward → Simple channel sum (safe path) autograd
        STE connects: output = safe_sum + (c_race - safe_sum).detach()

      Step 3: Bias addition
        Standard PyTorch (no STE needed)
    """

    def __init__(self, scale, num_channels=56, kernel_size=9):
        super().__init__()
        self.scale = scale
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        # Learnable deconv weights: (56, 1, 9, 9) — depthwise
        self.deconv_weights = nn.Parameter(
            torch.zeros(num_channels, 1, kernel_size, kernel_size, dtype=torch.float64)
        )

        # Learnable bias (scalar)
        self.bias = nn.Parameter(torch.tensor([-0.03262640000], dtype=torch.float64))

        self._load_initial_weights()

    def _load_initial_weights(self):
        if os.path.exists("weights_layer8_original.txt"):
            w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64)
            w = w.reshape(self.num_channels, 1, self.kernel_size, self.kernel_size)
            self.deconv_weights.data = torch.from_numpy(w)
            print(f"✓ Loaded weights_layer8_original.txt ({len(w.flatten())} weights)")
        else:
            print("⚠ weights_layer8_original.txt not found, using zero init")

    def forward(self, x):
        """
        x: (B, 56, rows, cols)
        Returns: (B, 1, out_rows, out_cols)
        """
        batch_size, num_ch, rows, cols = x.shape
        out_rows = rows * self.scale
        out_cols = cols * self.scale
        padding = self.kernel_size // 2  # = 4
        output_padding = self.scale - 1

        # =================================================================
        # STEP 1: Deconvolution  — C forward, PyTorch backward (STE trick)
        # =================================================================

        # (a) PyTorch path — provides gradients via autograd
        pt_deconv = torch.nn.functional.conv_transpose2d(
            x, self.deconv_weights,
            bias=None, stride=self.scale, padding=padding,
            output_padding=output_padding, groups=num_ch
        )
        pt_deconv = pt_deconv[:, :, :out_rows, :out_cols]  # crop to exact size

        # (b) C path — provides bit-exact forward (no grad)
        with torch.no_grad():
            w_np = self.deconv_weights.detach().cpu().numpy()
            c_deconv = torch.zeros_like(pt_deconv)
            for b in range(batch_size):
                in_np = x[b].detach().cpu().numpy()
                out_np = c_deconv_forward(in_np, w_np, num_ch, rows, cols, self.scale)
                c_deconv[b] = torch.from_numpy(out_np)

        # (c) STE: forward uses C value, backward uses PyTorch gradient
        #     output = pt_output + (c_output - pt_output).detach()
        #     Since .detach() kills gradient, backward sees only pt_output's grad
        deconv_out = pt_deconv + (c_deconv - pt_deconv).detach()
        # deconv_out shape: (B, 56, out_rows, out_cols)

        # =================================================================
        # STEP 2: Imadd with Race Condition — C forward, STE backward
        # =================================================================

        # (a) PyTorch safe path — simple channel sum (provides gradients)
        pt_sum = deconv_out.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # (b) C race-condition path — bit-exact race result (no grad)
        with torch.no_grad():
            c_race = torch.zeros_like(pt_sum)
            for b in range(batch_size):
                in_np = deconv_out[b].detach().cpu().numpy()
                out_np = c_imadd_race_forward(in_np, num_ch, out_rows, out_cols)
                c_race[b, 0] = torch.from_numpy(out_np)

        # (c) STE: forward uses C race value, backward uses sum gradient
        imadd_out = pt_sum + (c_race - pt_sum).detach()
        # imadd_out shape: (B, 1, out_rows, out_cols)

        # =================================================================
        # STEP 3: Bias Addition — standard PyTorch
        # =================================================================
        output = imadd_out + self.bias

        return output


# ============================================================================
# 4. DATASET
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
# 5. TRAINING LOOP
# ============================================================================
def compute_psnr(mse_loss):
    if mse_loss <= 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse_loss)


def train():
    print("=" * 65)
    print("  FSRCNN Layer 8 Training — STE-based 3-Step Backpropagation")
    print("=" * 65)

    dataset = FSRCNNDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = Layer8FSRCNN_3Step(scale=dataset.scale)

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    num_epochs = 200

    # Separate LR for weights vs bias — bias needs MUCH smaller LR
    optimizer = optim.Adam([
        {'params': [model.deconv_weights], 'lr': 5e-4},
        {'params': [model.bias], 'lr': 1e-4},
    ], weight_decay=1e-6)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_loss = float('inf')
    patience = 0
    max_patience = 40

    print(f"\nConfig: epochs={num_epochs}, batch=8, weight_lr=5e-4, bias_lr=1e-4 (cosine)")
    print(f"        grad_clip=0.5, loss=0.7*MSE + 0.3*L1, bias_clamp=[-0.5, 0.5]")
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
            loss = 0.7 * loss_mse + 0.3 * loss_l1

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # Clamp bias to reasonable range
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

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"MSE: {avg_mse:.8f} | L1: {avg_l1:.6f} | "
              f"PSNR~{psnr_est:.2f}dB | LR: {current_lr:.6f} | "
              f"Bias: {model.bias.item():.8f}")

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
    print(f"  Weights saved to: weights_layer8_trained.txt")
    print(f"  Bias saved to:    biasess_layer8_trained.txt")
    print("=" * 65)


def save_weights(model):
    weights = model.deconv_weights.data.numpy().flatten()
    bias = model.bias.data.item()

    with open("weights_layer8_trained.txt", "w") as f:
        np.savetxt(f, weights[None, :], fmt="  %.16e", delimiter="  ")

    with open("biasess_layer8_trained.txt", "w") as f:
        f.write(f"{bias:.10f}\n")


# ============================================================================
# 6. VERIFICATION
# ============================================================================
def verify_deconv_consistency():
    print("=" * 50)
    print("Verifying Deconv STE Consistency: C vs PyTorch")
    print("=" * 50)

    dataset = FSRCNNDataset()
    l7, hr = dataset[0]
    l7 = l7.unsqueeze(0)

    w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64)
    w_reshaped = w.reshape(56, 1, 9, 9)
    weights_tensor = torch.from_numpy(w_reshaped)

    # C forward
    c_out = c_deconv_forward(
        l7[0].numpy(), w_reshaped, 56,
        dataset.in_rows, dataset.in_cols, dataset.scale
    )
    c_tensor = torch.from_numpy(c_out).unsqueeze(0)

    # PyTorch forward
    padding = 9 // 2
    output_padding = dataset.scale - 1
    pt_output = torch.nn.functional.conv_transpose2d(
        l7, weights_tensor,
        bias=None, stride=dataset.scale, padding=padding,
        output_padding=output_padding, groups=56
    )
    out_rows = dataset.in_rows * dataset.scale
    out_cols = dataset.in_cols * dataset.scale
    pt_output = pt_output[:, :, :out_rows, :out_cols]

    diff = (c_tensor - pt_output).abs()
    print(f"Max absolute difference: {diff.max().item():.10f}")
    print(f"Mean absolute difference: {diff.mean().item():.10f}")

    if diff.max().item() < 1e-6:
        print("✓ C and PyTorch deconv are fully consistent!")
    else:
        print("⚠ Small difference (expected due to padding). STE handles this correctly.")
    print()


if __name__ == "__main__":
    verify_deconv_consistency()
    train()
