#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning FSRCNN v4: Hybrid (Channel Equalization + Noise Injection)

Menggabungkan v3 (channel equalization) dan v1 (noise injection):
- Channel equalization: meratakan kontribusi per-channel deconv
- Noise injection: injeksi noise empiris dari train_data/ saat training
- Freeze layer 1-7, hanya train Layer 8

Loss = MSE(output + noise, GT) + λ_eq * var(channel_energies)
                               + λ_reg * ||W - W_original||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

# ==================== Konfigurasi ====================
CONFIG = {
    'lr_width': 176,
    'lr_height': 144,
    'scale': 2,
    'num_frames': 150,
    'num_runs': 30,
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-5,
    'checkpoint_interval': 20,
    'lambda_eq': 0.1,          # channel equalization
    'lambda_reg': 0.005,       # L2 regularization ke original
    'noise_probability': 0.5,  # probabilitas injeksi noise per batch
    'data_root': 'train_data',
    'lr_video': 'suzie_qcif.yuv',
    'weights_in': 'fsrcnn_original.pth',
    'weights_out': 'fsrcnn_finetuned_v4.pth',
}


# ==================== Model ====================
class PReLU_custom(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        return torch.where(x > 0, x, self.coeff * x)


class FSRCNNv4(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
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
        self.deconv8 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2,
                                           padding=4, output_padding=1)

    def forward_features(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))
        x = self.prelu7(self.conv7(x))
        return x

    def forward_deconv_per_channel(self, features):
        B, C, H, W = features.shape
        weight = self.deconv8.weight
        bias = self.deconv8.bias
        per_channel = []
        for j in range(C):
            ch_input = features[:, j:j+1, :, :]
            ch_weight = weight[j:j+1, :, :, :]
            ch_out = F.conv_transpose2d(ch_input, ch_weight,
                                         stride=self.scale,
                                         padding=4, output_padding=1)
            per_channel.append(ch_out)
        total = torch.stack(per_channel, dim=0).sum(dim=0)
        total = total + bias.view(1, 1, 1, 1)
        return per_channel, total

    def forward(self, x):
        features = self.forward_features(x)
        return self.deconv8(features)


# ==================== Dataset ====================
class YUVDatasetWithNoise(Dataset):
    """Dataset dengan noise pool dari train_data/."""
    def __init__(self, lr_video_path, gt_dir, noise_dir, config):
        self.lr_shape = (config['lr_height'], config['lr_width'])
        self.hr_shape = (config['lr_height'] * config['scale'],
                         config['lr_width'] * config['scale'])

        print("Memuat frame LR...")
        self.lr_frames = self._load_lr(lr_video_path, config['num_frames'])
        print(f"  {len(self.lr_frames)} frame LR")

        print("Memuat frame GT...")
        self.gt_frames = self._load_gt(gt_dir, config['num_frames'])
        print(f"  {len(self.gt_frames)} frame GT")

        print("Membangun noise pool...")
        self.noise_pool = self._build_noise_pool(noise_dir, gt_dir, config)
        print(f"  {len(self.noise_pool)} sampel noise")

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

    def _build_noise_pool(self, noise_base, gt_dir, config):
        """Hitung noise = runX - gt untuk setiap frame."""
        noise_pool = []
        h, w = self.hr_shape
        for run_idx in range(config['num_runs']):
            run_dir = os.path.join(noise_base, f'run{run_idx}')
            if not os.path.isdir(run_dir): continue
            for frame_idx in range(config['num_frames']):
                run_path = os.path.join(run_dir, f'frame_{frame_idx:04d}.yuv')
                gt_path = os.path.join(gt_dir, f'frame_{frame_idx:04d}.yuv')
                if not os.path.exists(run_path) or not os.path.exists(gt_path):
                    continue
                run_data = np.fromfile(run_path, dtype=np.uint8)
                gt_data = np.fromfile(gt_path, dtype=np.uint8)
                if len(run_data) >= h*w and len(gt_data) >= h*w:
                    run_y = run_data[:h*w].reshape(h, w).astype(np.float32) / 255.0
                    gt_y = gt_data[:h*w].reshape(h, w).astype(np.float32) / 255.0
                    noise = run_y - gt_y
                    if np.abs(noise).max() > 1e-6:
                        noise_pool.append(noise)
        # Fallback jika tidak cukup noise
        if len(noise_pool) < 5:
            print("  ⚠ Noise empiris kurang, generate Gaussian (std=0.012)")
            for _ in range(50):
                noise_pool.append(np.random.normal(0, 0.012, (h, w)).astype(np.float32))
        return noise_pool

    def __len__(self):
        return min(len(self.lr_frames), len(self.gt_frames))

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_frames[idx]).unsqueeze(0).float()
        gt = torch.from_numpy(self.gt_frames[idx]).unsqueeze(0).float()
        # Random noise dari pool
        noise_idx = random.randint(0, len(self.noise_pool) - 1)
        noise = torch.from_numpy(self.noise_pool[noise_idx]).unsqueeze(0).float()
        return lr, gt, noise


# ==================== Utilitas ====================
def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10: return 100.0
    return 10.0 * np.log10(1.0 / mse)


def channel_equalization_loss(per_channel_outputs):
    energies = []
    for ch_out in per_channel_outputs:
        energies.append(torch.mean(ch_out ** 2))
    energies = torch.stack(energies)
    return torch.var(energies)


def export_weights_to_txt(model, suffix='_finetuned_v4'):
    print(f"\nMengekspor bobot ke file *{suffix}.txt ...")
    layers = {
        'conv1': 1, 'conv2': 2, 'conv3': 3, 'conv4': 4,
        'conv5': 5, 'conv6': 6, 'conv7': 7, 'deconv8': 8,
    }
    for attr, num in layers.items():
        layer = getattr(model, attr)
        w = layer.weight.data.cpu().numpy().flatten()
        b = layer.bias.data.cpu().numpy().flatten()
        with open(f'weights_layer{num}{suffix}.txt', 'w') as f:
            for val in w: f.write(f'{val:.10f}\n')
        with open(f'biasess_layer{num}{suffix}.txt', 'w') as f:
            for val in b: f.write(f'{val:.10f}\n')
        changed = " ← CHANGED" if attr == 'deconv8' else ""
        print(f"  ✅ Layer {num}: {len(w)}w + {len(b)}b{changed}")
    print("Export selesai!")


# ==================== Training ====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load model
    model = FSRCNNv4(scale=CONFIG['scale']).to(device)
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable} params (deconv8 only)")

    # 3. Original weights (untuk regularisasi)
    orig_w = model.deconv8.weight.data.clone()
    orig_b = model.deconv8.bias.data.clone()

    # Channel analysis sebelum
    w_norms = [torch.norm(model.deconv8.weight[j]).item() for j in range(56)]
    print(f"\nChannel norms (sebelum): mean={np.mean(w_norms):.4f}, "
          f"std={np.std(w_norms):.4f}, ratio={max(w_norms)/min(w_norms):.1f}x")

    # 4. Dataset
    gt_dir = os.path.join(CONFIG['data_root'], 'gt')
    dataset = YUVDatasetWithNoise(CONFIG['lr_video'], gt_dir,
                                   CONFIG['data_root'], CONFIG)
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
        for lr, gt, _ in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                total_psnr += compute_psnr(out[i], gt[i]); count += 1
        baseline = total_psnr / count
        print(f"\nBaseline PSNR: {baseline:.2f} dB")

    # 7. Training
    print(f"\nTraining: {CONFIG['num_epochs']} epoch")
    print(f"  λ_eq={CONFIG['lambda_eq']}, λ_reg={CONFIG['lambda_reg']}, "
          f"noise_prob={CONFIG['noise_probability']}")
    print("=" * 70)

    best_combined_score = 0

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0; epoch_mse = 0; epoch_eq = 0
        num_batches = 0

        for lr, gt, noise in dataloader:
            lr, gt, noise = lr.to(device), gt.to(device), noise.to(device)
            optimizer.zero_grad()

            # Forward per-channel (untuk equalization loss)
            features = model.forward_features(lr)
            per_ch, output = model.forward_deconv_per_channel(features)

            # Noise injection dengan probabilitas
            if random.random() < CONFIG['noise_probability']:
                output_noisy = output + noise
                l_mse = mse_loss(output_noisy, gt)
            else:
                l_mse = mse_loss(output, gt)

            # Channel equalization
            l_eq = channel_equalization_loss(per_ch)

            # Weight regularization
            l_reg = (torch.mean((model.deconv8.weight - orig_w) ** 2) +
                     torch.mean((model.deconv8.bias - orig_b) ** 2))

            loss = l_mse + CONFIG['lambda_eq'] * l_eq + CONFIG['lambda_reg'] * l_reg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += l_mse.item()
            epoch_eq += l_eq.item()
            num_batches += 1

        n = max(num_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                psnr_clean = 0; psnr_noisy = 0; count = 0
                for lr_v, gt_v, noise_v in dataloader:
                    lr_v, gt_v = lr_v.to(device), gt_v.to(device)
                    noise_v = noise_v.to(device)
                    out = model(lr_v)
                    out_noisy = out + noise_v
                    for i in range(lr_v.size(0)):
                        psnr_clean += compute_psnr(out[i], gt_v[i])
                        psnr_noisy += compute_psnr(out_noisy[i], gt_v[i])
                        count += 1
                pc = psnr_clean / count
                pn = psnr_noisy / count

                w_norms = [torch.norm(model.deconv8.weight[j]).item() for j in range(56)]
                ratio = max(w_norms) / max(min(w_norms), 1e-10)

                print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                      f"Loss:{epoch_loss/n:.6f} (MSE:{epoch_mse/n:.6f} EQ:{epoch_eq/n:.6f})  "
                      f"PSNR clean:{pc:.2f} noisy:{pn:.2f}  "
                      f"ChRatio:{ratio:.1f}x")

                # Save best: balance PSNR noisy + channel ratio
                score = pn + pc * 0.3 - ratio * 0.1
                if score > best_combined_score or epoch == 0:
                    best_combined_score = score
                    torch.save(model.state_dict(), CONFIG['weights_out'])
        else:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                  f"Loss:{epoch_loss/n:.6f} (MSE:{epoch_mse/n:.6f} EQ:{epoch_eq/n:.6f})")

        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
            ckpt = f'fsrcnn_finetuned_v4_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 {ckpt}")

    # 8. Final
    print("=" * 70)
    model.load_state_dict(torch.load(CONFIG['weights_out'], map_location=device,
                                      weights_only=True))
    model.eval()

    w_norms = [torch.norm(model.deconv8.weight[j]).item() for j in range(56)]
    print(f"\nChannel norms (sesudah): mean={np.mean(w_norms):.4f}, "
          f"std={np.std(w_norms):.4f}, ratio={max(w_norms)/min(w_norms):.1f}x")

    with torch.no_grad():
        psnr_c = 0; psnr_n = 0; count = 0
        for lr, gt, noise in dataloader:
            lr, gt, noise = lr.to(device), gt.to(device), noise.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                psnr_c += compute_psnr(out[i], gt[i])
                psnr_n += compute_psnr(out[i] + noise[i], gt[i])
                count += 1
        print(f"Final PSNR clean: {psnr_c/count:.2f} dB")
        print(f"Final PSNR noisy: {psnr_n/count:.2f} dB")

    # 9. Export
    export_weights_to_txt(model, suffix='_finetuned_v4')

    print("\n🎉 Selesai!")
    print("  Untuk C code: ganti weights_layer8.txt → weights_layer8_finetuned_v4.txt")
    print("  Layer 1-7 TIDAK BERUBAH")


if __name__ == '__main__':
    train()