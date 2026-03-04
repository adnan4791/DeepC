#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Micro-Compensator: 8 features, 2 layers
Input (1ch) → Conv 3×3 (1→8) → ReLU → Conv 3×3 (8→1) → Output
Total: 217 parameter — sangat ringan untuk C inference

Dilatih pada data noisy (race condition) vs clean (serial/critical).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

# ==================== Konfigurasi ====================
CONFIG = {
    'hr_width': 352,
    'hr_height': 288,
    'num_frames': 150,
    'num_runs': 30,        # jumlah run di train_data/
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'num_features': 8,     # KECIL! 8 features saja
    'data_root': 'train_data',
    'lr_video': 'suzie_qcif.yuv',
}

H = CONFIG['hr_height']
W = CONFIG['hr_width']


# ==================== Model ====================
class MicroCompensator(nn.Module):
    """2-layer CNN, 8 features. Total ~217 params."""
    def __init__(self, num_features=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


# ==================== Dataset ====================
class NoisyCleanDataset(Dataset):
    """Pasangkan frame noisy (dari parallel run) dengan clean (dari serial)."""
    def __init__(self, data_root, config):
        self.H = config['hr_height']
        self.W = config['hr_width']
        frame_size = self.H * self.W

        gt_dir = os.path.join(data_root, 'gt')

        # Muat frame GT (clean)
        print("Memuat frame GT (clean)...")
        self.gt_frames = []
        for i in range(config['num_frames']):
            p = os.path.join(gt_dir, f'frame_{i:04d}.yuv')
            if not os.path.exists(p):
                break
            d = np.fromfile(p, dtype=np.uint8)
            if len(d) >= frame_size:
                self.gt_frames.append(
                    d[:frame_size].reshape(self.H, self.W).astype(np.float32) / 255.0
                )
        print(f"  {len(self.gt_frames)} frame GT")

        # Muat frame noisy dari semua run
        print("Memuat frame noisy (dari parallel runs)...")
        self.pairs = []  # list of (noisy_frame, gt_index)
        for run_idx in range(config['num_runs']):
            run_dir = os.path.join(data_root, f'run{run_idx}')
            if not os.path.isdir(run_dir):
                continue
            for frame_idx in range(len(self.gt_frames)):
                p = os.path.join(run_dir, f'frame_{frame_idx:04d}.yuv')
                if not os.path.exists(p):
                    continue
                d = np.fromfile(p, dtype=np.uint8)
                if len(d) >= frame_size:
                    noisy = d[:frame_size].reshape(self.H, self.W).astype(np.float32) / 255.0
                    self.pairs.append((noisy, frame_idx))
        print(f"  {len(self.pairs)} pasangan noisy-clean")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, gt_idx = self.pairs[idx]
        clean = self.gt_frames[gt_idx]

        # Random crop 64×64 untuk augmentasi
        ps = 64
        i = random.randint(0, self.H - ps - 1)
        j = random.randint(0, self.W - ps - 1)
        noisy_patch = noisy[i:i+ps, j:j+ps]
        clean_patch = clean[i:i+ps, j:j+ps]

        return (torch.from_numpy(noisy_patch).unsqueeze(0).float(),
                torch.from_numpy(clean_patch).unsqueeze(0).float())


# ==================== Utilitas ====================
def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def export_micro_weights(model, filename='micro_compensator_weights.txt'):
    """Export bobot ke format teks untuk C."""
    print(f"\nExport bobot ke {filename}...")
    with open(filename, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f'# {name} shape={list(param.shape)}\n')
            vals = param.data.cpu().numpy().flatten()
            for v in vals:
                f.write(f'{v:.10f}\n')
            f.write('\n')
    
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total} parameter")
    
    # Detail per layer
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel()} vals")
    print("Export selesai!")


# ==================== Training ====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    nf = CONFIG['num_features']
    model = MicroCompensator(num_features=nf).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MicroCompensator: {nf} features, {total_params} params")

    dataset = NoisyCleanDataset(CONFIG['data_root'], CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True, num_workers=0)

    criterion = nn.L1Loss()  # L1 lebih robust untuk denoising
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    best_loss = float('inf')

    print(f"\nTraining: {CONFIG['num_epochs']} epoch")
    print("=" * 60)

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                psnr_noisy = 0
                psnr_comp = 0
                count = 0
                for noisy, clean in dataloader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    output = model(noisy)
                    for i in range(noisy.size(0)):
                        psnr_noisy += compute_psnr(noisy[i], clean[i])
                        psnr_comp += compute_psnr(output[i], clean[i])
                        count += 1
                pn = psnr_noisy / count
                pc = psnr_comp / count
                print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                      f"Loss:{avg_loss:.6f}  "
                      f"PSNR noisy:{pn:.2f} → compensated:{pc:.2f}  "
                      f"(+{pc-pn:.2f} dB)")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), 'micro_compensator_best.pth')
        else:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  Loss:{avg_loss:.6f}")

    # Final
    print("=" * 60)
    model.load_state_dict(torch.load('micro_compensator_best.pth',
                                      map_location=device, weights_only=True))
    export_micro_weights(model, 'micro_compensator_weights.txt')
    torch.save(model.state_dict(), 'micro_compensator_final.pth')
    print("\n🎉 Selesai! File output:")
    print("  - micro_compensator_best.pth")
    print("  - micro_compensator_weights.txt (untuk C code)")


if __name__ == '__main__':
    train()
