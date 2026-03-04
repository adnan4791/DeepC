#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning FSRCNN v3: Channel Equalization

Strategi: Meratakan kontribusi per-channel di deconv Layer 8
agar hilangnya 1-2 channel karena race condition memiliki dampak minimal.

Loss = MSE(output, GT) + λ_eq * var(channel_contributions)
                       + λ_reg * ||W - W_original||²

Freeze layer 1-7, hanya train Layer 8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==================== Konfigurasi ====================
CONFIG = {
    'lr_width': 176,
    'lr_height': 144,
    'scale': 2,
    'num_frames': 150,
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-5,
    'checkpoint_interval': 20,
    'lambda_eq': 0.1,          # bobot channel equalization loss
    'lambda_reg': 0.005,       # bobot L2 regularization ke original
    'data_root': 'train_data',
    'lr_video': 'suzie_qcif.yuv',
    'weights_in': 'fsrcnn_original.pth',
    'weights_out': 'fsrcnn_finetuned_v3.pth',
}


# ==================== Model ====================
class PReLU_custom(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        return torch.where(x > 0, x, self.coeff * x)


class FSRCNNv3(nn.Module):
    """
    FSRCNN dengan eksplisit per-channel deconv output untuk channel equalization.
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        # Layer 1-7 (frozen)
        self.conv1 = nn.Conv2d(1, 56, kernel_size=5, padding=2)
        self.prelu1 = PReLU_custom(-0.8986)
        self.conv2 = nn.Conv2d(56, 12, kernel_size=1)
        self.prelu2 = PReLU_custom(0.3236)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu3 = PReLU_custom(0.2288)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu4 = PReLU_custom(0.2476)
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu5 = PReLU_custom(0.3495)
        self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu6 = PReLU_custom(0.7806)
        self.conv7 = nn.Conv2d(12, 56, kernel_size=1)
        self.prelu7 = PReLU_custom(0.0087)
        # Layer 8 (trainable)
        self.deconv8 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2,
                                           padding=4, output_padding=1)

    def forward_features(self, x):
        """Forward layer 1-7, return feature map (B, 56, H, W)."""
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))
        x = self.prelu7(self.conv7(x))
        return x  # (B, 56, H_lr, W_lr)

    def forward_deconv_per_channel(self, features):
        """
        Hitung deconv per channel secara eksplisit.
        Return: (per_channel_outputs, total_output)
          per_channel_outputs: list of 56 tensors, masing-masing (B, 1, H_hr, W_hr)
          total_output: sum of all channels + bias, (B, 1, H_hr, W_hr)
        """
        B, C, H, W = features.shape  # C=56
        weight = self.deconv8.weight  # shape: (56, 1, 9, 9) — in_channels, out_channels, kH, kW
        bias = self.deconv8.bias      # shape: (1,)

        per_channel = []
        for j in range(C):
            # Ambil channel j: (B, 1, H, W)
            ch_input = features[:, j:j+1, :, :]
            # Ambil weight channel j: (1, 1, 9, 9)
            ch_weight = weight[j:j+1, :, :, :]
            # Deconv tanpa bias
            ch_out = F.conv_transpose2d(ch_input, ch_weight,
                                         stride=self.scale,
                                         padding=4, output_padding=1)
            per_channel.append(ch_out)

        # Total = sum semua channel + bias
        total = torch.stack(per_channel, dim=0).sum(dim=0)  # (B, 1, H_hr, W_hr)
        total = total + bias.view(1, 1, 1, 1)

        return per_channel, total

    def forward(self, x):
        """Standard forward pass (untuk inference)."""
        features = self.forward_features(x)
        x = self.deconv8(features)
        return x


# ==================== Dataset ====================
class YUVDataset(Dataset):
    def __init__(self, lr_video_path, gt_dir, config):
        self.lr_shape = (config['lr_height'], config['lr_width'])
        self.hr_shape = (config['lr_height'] * config['scale'],
                         config['lr_width'] * config['scale'])

        print("Memuat frame LR & GT...")
        self.lr_frames = self._load_lr(lr_video_path, config['num_frames'])
        self.gt_frames = self._load_gt(gt_dir, config['num_frames'])
        print(f"  {len(self.lr_frames)} LR, {len(self.gt_frames)} GT")

    def _load_lr(self, path, num_frames):
        frames = []
        h, w = self.lr_shape
        y_size = h * w
        frame_size = y_size + 2 * (h//2 * w//2)
        with open(path, 'rb') as f:
            for _ in range(num_frames):
                raw = f.read(frame_size)
                if len(raw) < frame_size: break
                frames.append(np.frombuffer(raw[:y_size], dtype=np.uint8
                              ).reshape(h, w).astype(np.float32) / 255.0)
        return frames

    def _load_gt(self, folder, num_frames):
        frames = []
        h, w = self.hr_shape
        for i in range(num_frames):
            p = os.path.join(folder, f'frame_{i:04d}.yuv')
            if not os.path.exists(p): break
            d = np.fromfile(p, dtype=np.uint8)
            if len(d) >= h*w:
                frames.append(d[:h*w].reshape(h, w).astype(np.float32) / 255.0)
        return frames

    def __len__(self):
        return min(len(self.lr_frames), len(self.gt_frames))

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_frames[idx]).unsqueeze(0).float()
        gt = torch.from_numpy(self.gt_frames[idx]).unsqueeze(0).float()
        return lr, gt


# ==================== Utilitas ====================
def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10: return 100.0
    return 10.0 * np.log10(1.0 / mse)


def channel_equalization_loss(per_channel_outputs):
    """
    Hitung variance kontribusi per-channel.
    Setiap channel menghasilkan output (B, 1, H, W).
    Kita hitung L2 norm rata-rata per channel, lalu variance-nya.
    Semakin merata → variance semakin kecil.
    """
    # Hitung energy (L2 norm²) per channel, rata-rata atas batch dan piksel
    energies = []
    for ch_out in per_channel_outputs:
        energy = torch.mean(ch_out ** 2)  # scalar
        energies.append(energy)
    energies = torch.stack(energies)  # (56,)
    # Variance energi antar channel (ingin diminimalkan)
    var_loss = torch.var(energies)
    return var_loss


def export_weights_to_txt(model, output_dir='.', suffix='_finetuned_v3'):
    """Export bobot ke .txt kompatibel C code."""
    print(f"\nMengekspor bobot ke file *{suffix}.txt ...")
    layers = {
        'conv1': (1, 'weights_layer1', 'biasess_layer1'),
        'conv2': (2, 'weights_layer2', 'biasess_layer2'),
        'conv3': (3, 'weights_layer3', 'biasess_layer3'),
        'conv4': (4, 'weights_layer4', 'biasess_layer4'),
        'conv5': (5, 'weights_layer5', 'biasess_layer5'),
        'conv6': (6, 'weights_layer6', 'biasess_layer6'),
        'conv7': (7, 'weights_layer7', 'biasess_layer7'),
        'deconv8': (8, 'weights_layer8', 'biasess_layer8'),
    }
    for attr, (num, w_prefix, b_prefix) in layers.items():
        layer = getattr(model, attr)
        w = layer.weight.data.cpu().numpy().flatten()
        b = layer.bias.data.cpu().numpy().flatten()
        w_path = os.path.join(output_dir, f'{w_prefix}{suffix}.txt')
        b_path = os.path.join(output_dir, f'{b_prefix}{suffix}.txt')
        with open(w_path, 'w') as f:
            for val in w: f.write(f'{val:.10f}\n')
        with open(b_path, 'w') as f:
            for val in b: f.write(f'{val:.10f}\n')
        changed = " (CHANGED)" if attr == 'deconv8' else ""
        print(f"  ✅ Layer {num}: {len(w)} weights, {len(b)} biases{changed}")
    print("Export selesai!")


# ==================== Training ====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load model
    model = FSRCNNv3(scale=CONFIG['scale']).to(device)
    if os.path.exists(CONFIG['weights_in']):
        model.load_state_dict(torch.load(CONFIG['weights_in'], map_location=device,
                                          weights_only=True))
        print(f"✅ Bobot dimuat: {CONFIG['weights_in']}")
    else:
        print(f"❌ {CONFIG['weights_in']} tidak ditemukan!"); return

    # 2. Freeze layer 1-7
    for name, param in model.named_parameters():
        if not name.startswith('deconv8'):
            param.requires_grad = False
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 3. Simpan bobot original deconv8
    orig_w = model.deconv8.weight.data.clone()
    orig_b = model.deconv8.bias.data.clone()

    # Analisis channel energy sebelum training
    print("\n--- Analisis Channel Energy (sebelum) ---")
    w_norms = []
    for j in range(56):
        norm = torch.norm(model.deconv8.weight[j]).item()
        w_norms.append(norm)
    w_norms = np.array(w_norms)
    print(f"  Weight norm per channel: mean={w_norms.mean():.4f}, "
          f"std={w_norms.std():.4f}, min={w_norms.min():.4f}, max={w_norms.max():.4f}")
    print(f"  Ratio max/min: {w_norms.max()/w_norms.min():.2f}x")

    # 4. Dataset
    dataset = YUVDataset(CONFIG['lr_video'],
                          os.path.join(CONFIG['data_root'], 'gt'), CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True, num_workers=0)

    # 5. Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG['learning_rate'])
    mse_loss = nn.MSELoss()

    # 6. Baseline
    model.eval()
    with torch.no_grad():
        total_psnr = 0; count = 0
        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                total_psnr += compute_psnr(out[i], gt[i]); count += 1
        baseline_psnr = total_psnr / count
        print(f"\nBaseline PSNR: {baseline_psnr:.2f} dB")

    # 7. Training
    print(f"\nTraining: {CONFIG['num_epochs']} epoch")
    print(f"  λ_eq={CONFIG['lambda_eq']}, λ_reg={CONFIG['lambda_reg']}")
    print("=" * 65)

    best_score = 0  # combined score: psnr - penalty

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0; epoch_mse = 0; epoch_eq = 0; epoch_reg = 0
        num_batches = 0

        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            optimizer.zero_grad()

            # Forward dengan per-channel output
            features = model.forward_features(lr)
            per_ch, output = model.forward_deconv_per_channel(features)

            # Losses
            l_mse = mse_loss(output, gt)
            l_eq = channel_equalization_loss(per_ch)
            l_reg = (torch.mean((model.deconv8.weight - orig_w) ** 2) +
                     torch.mean((model.deconv8.bias - orig_b) ** 2))

            loss = l_mse + CONFIG['lambda_eq'] * l_eq + CONFIG['lambda_reg'] * l_reg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += l_mse.item()
            epoch_eq += l_eq.item()
            epoch_reg += l_reg.item()
            num_batches += 1

        n = max(num_batches, 1)
        avg_loss = epoch_loss / n
        avg_mse = epoch_mse / n
        avg_eq = epoch_eq / n
        avg_reg = epoch_reg / n

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                total_psnr = 0; count = 0
                for lr_v, gt_v in dataloader:
                    lr_v, gt_v = lr_v.to(device), gt_v.to(device)
                    out = model(lr_v)
                    for i in range(lr_v.size(0)):
                        total_psnr += compute_psnr(out[i], gt_v[i]); count += 1
                psnr = total_psnr / count

                # Channel energy analysis
                w_norms = [torch.norm(model.deconv8.weight[j]).item() for j in range(56)]
                w_std = np.std(w_norms)
                w_ratio = max(w_norms) / max(min(w_norms), 1e-10)

                print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                      f"Loss: {avg_loss:.6f} (MSE:{avg_mse:.6f} EQ:{avg_eq:.6f} REG:{avg_reg:.6f})  "
                      f"PSNR: {psnr:.2f} dB  "
                      f"ChNorm std:{w_std:.4f} ratio:{w_ratio:.1f}x")

                # Save best: balance antara PSNR dan equalization
                score = psnr - 0.5 * w_ratio  # bonus untuk channel lebih merata
                if score > best_score or epoch == 0:
                    best_score = score
                    torch.save(model.state_dict(), CONFIG['weights_out'])
        else:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                  f"Loss: {avg_loss:.6f} (MSE:{avg_mse:.6f} EQ:{avg_eq:.6f})")

        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
            ckpt = f'fsrcnn_finetuned_v3_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 {ckpt}")

    # 8. Final
    print("=" * 65)
    model.load_state_dict(torch.load(CONFIG['weights_out'], map_location=device,
                                      weights_only=True))
    model.eval()

    # Final channel analysis
    print("\n--- Analisis Channel Energy (sesudah) ---")
    w_norms = [torch.norm(model.deconv8.weight[j]).item() for j in range(56)]
    w_norms = np.array(w_norms)
    print(f"  Weight norm per channel: mean={w_norms.mean():.4f}, "
          f"std={w_norms.std():.4f}, min={w_norms.min():.4f}, max={w_norms.max():.4f}")
    print(f"  Ratio max/min: {w_norms.max()/w_norms.min():.2f}x")

    with torch.no_grad():
        total_psnr = 0; count = 0
        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                total_psnr += compute_psnr(out[i], gt[i]); count += 1
        print(f"\nFinal PSNR: {total_psnr/count:.2f} dB")

    # 9. Export
    export_weights_to_txt(model, suffix='_finetuned_v3')

    print("\n🎉 Selesai!")
    print("  Untuk C code: ganti weights_layer8.txt dengan weights_layer8_finetuned_v3.txt")
    print("  Layer 1-7 tetap sama (tidak berubah)")


if __name__ == '__main__':
    train()