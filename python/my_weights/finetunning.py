#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning FSRCNN untuk Race Condition Robustness.

Strategi: Noise-Aware Training
- Load bobot original dari fsrcnn_original.pth
- Baca noise empiris dari train_data/ (runX - gt)
- Saat training, injeksikan noise ke output deconv layer
- Model belajar menghasilkan bobot yang toleran terhadap race condition
- Export bobot ke .txt untuk C code
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import random

# ==================== Konfigurasi ====================
CONFIG = {
    'lr_width': 176,
    'lr_height': 144,
    'scale': 2,
    'num_frames': 150,
    'num_runs': 30,
    'batch_size': 4,
    'num_epochs': 30,
    'learning_rate': 1e-5,
    'checkpoint_interval': 5,
    'noise_probability': 0.7,  # probabilitas injeksi noise per batch
    'data_root': 'train_data',
    'lr_video': 'suzie_qcif.yuv',
    'weights_in': 'fsrcnn_original.pth',
    'weights_out': 'fsrcnn_finetuned.pth',
}

# ==================== Model FSRCNN ====================
class PReLU_custom(nn.Module):
    """PReLU dengan koefisien tetap (tidak trainable), sesuai C code."""
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.where(x > 0, x, self.coeff * x)


class FSRCNN(nn.Module):
    """
    FSRCNN dengan arsitektur identik dengan C code:
    Layer 1: Conv2d(1, 56, 5x5)  + PReLU(-0.8986)
    Layer 2: Conv2d(56, 12, 1x1) + PReLU(0.3236)
    Layer 3: Conv2d(12, 12, 3x3) + PReLU(0.2288)
    Layer 4: Conv2d(12, 12, 3x3) + PReLU(0.2476)
    Layer 5: Conv2d(12, 12, 3x3) + PReLU(0.3495)
    Layer 6: Conv2d(12, 12, 3x3) + PReLU(0.7806)
    Layer 7: Conv2d(12, 56, 1x1) + PReLU(0.0087)
    Layer 8: ConvTranspose2d(56, 1, 9x9, stride=2)
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        # Layer 1: Feature Extraction
        self.conv1 = nn.Conv2d(1, 56, kernel_size=5, padding=2)
        self.prelu1 = PReLU_custom(-0.8986)
        # Layer 2: Shrinking
        self.conv2 = nn.Conv2d(56, 12, kernel_size=1)
        self.prelu2 = PReLU_custom(0.3236)
        # Layer 3-6: Non-linear Mapping
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu3 = PReLU_custom(0.2288)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu4 = PReLU_custom(0.2476)
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu5 = PReLU_custom(0.3495)
        self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.prelu6 = PReLU_custom(0.7806)
        # Layer 7: Expanding
        self.conv7 = nn.Conv2d(12, 56, kernel_size=1)
        self.prelu7 = PReLU_custom(0.0087)
        # Layer 8: Deconvolution (Upsampling)
        self.deconv8 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2,
                                           padding=4, output_padding=1)

    def forward(self, x, noise_map=None):
        """
        Forward pass.
        noise_map: jika diberikan, ditambahkan ke output deconv sebelum
                   bias (simulasi race condition di Layer 8 imadd).
                   Shape: (B, 1, H_hr, W_hr)
        """
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))
        x = self.prelu7(self.conv7(x))
        x = self.deconv8(x)
        # Injeksi noise (simulasi race condition di imadd Layer 8)
        if noise_map is not None:
            x = x + noise_map
        return x


# ==================== Dataset ====================
class RaceConditionDataset(Dataset):
    """
    Dataset untuk noise-aware training.
    - LR frames dibaca dari suzie_qcif.yuv (Y-only, uint8)
    - HR ground truth dibaca dari train_data/gt/
    - Noise pool dihitung dari (runX - gt) pada stage precompute
    """
    def __init__(self, lr_video_path, data_root, config):
        self.config = config
        hr_h = config['lr_height'] * config['scale']
        hr_w = config['lr_width'] * config['scale']
        self.hr_shape = (hr_h, hr_w)
        self.lr_shape = (config['lr_height'], config['lr_width'])

        print("Memuat frame LR dari YUV...")
        self.lr_frames = self._load_lr_frames(lr_video_path)
        print(f"  Dimuat {len(self.lr_frames)} frame LR")

        print("Memuat frame GT (ground truth)...")
        self.gt_frames = self._load_hr_frames(
            os.path.join(data_root, 'gt'), config['num_frames'])
        print(f"  Dimuat {len(self.gt_frames)} frame GT")

        print("Membangun noise pool dari 30 run...")
        self.noise_pool = self._build_noise_pool(data_root, config)
        print(f"  Noise pool: {len(self.noise_pool)} sampel noise")

    def _load_lr_frames(self, path):
        """Baca komponen Y dari file YUV 4:2:0."""
        frames = []
        h, w = self.lr_shape
        y_size = h * w
        uv_size = (h // 2) * (w // 2)
        frame_size = y_size + 2 * uv_size  # Y + U + V

        with open(path, 'rb') as f:
            for i in range(self.config['num_frames']):
                raw = f.read(frame_size)
                if len(raw) < frame_size:
                    print(f"  Hanya bisa membaca {i} frame dari {path}")
                    break
                y_data = np.frombuffer(raw[:y_size], dtype=np.uint8)
                frame = y_data.reshape(h, w).astype(np.float32) / 255.0
                frames.append(frame)
        return frames

    def _load_hr_frames(self, folder, num_frames):
        """Baca frame HR dari file .yuv individual (Y-only, uint8)."""
        frames = []
        h, w = self.hr_shape
        for i in range(num_frames):
            path = os.path.join(folder, f'frame_{i:04d}.yuv')
            if not os.path.exists(path):
                break
            data = np.fromfile(path, dtype=np.uint8)
            if len(data) < h * w:
                print(f"  Frame {path} terlalu kecil, skip")
                continue
            frame = data[:h * w].reshape(h, w).astype(np.float32) / 255.0
            frames.append(frame)
        return frames

    def _build_noise_pool(self, data_root, config):
        """
        Hitung noise = runX - gt untuk setiap frame dan setiap run.
        Simpan sebagai list of numpy arrays (H_hr x W_hr).
        """
        noise_pool = []
        h, w = self.hr_shape
        num_runs = config['num_runs']
        num_frames = min(config['num_frames'], len(self.gt_frames))

        for frame_idx in range(num_frames):
            gt = self.gt_frames[frame_idx]
            for run_idx in range(num_runs):
                run_path = os.path.join(
                    data_root, f'run{run_idx}', f'frame_{frame_idx:04d}.yuv')
                if not os.path.exists(run_path):
                    continue
                data = np.fromfile(run_path, dtype=np.uint8)
                if len(data) < h * w:
                    continue
                run_frame = data[:h * w].reshape(h, w).astype(np.float32) / 255.0
                noise = run_frame - gt
                # Hanya simpan noise yang non-zero (ada race condition)
                if np.any(noise != 0):
                    noise_pool.append(noise)

        if len(noise_pool) == 0:
            print("  PERINGATAN: Noise pool kosong! Menggunakan noise Gaussian.")
            # Fallback: buat noise Gaussian berdasarkan statistik yang diketahui
            for _ in range(100):
                noise_pool.append(
                    np.random.normal(0, 0.012, (h, w)).astype(np.float32))

        return noise_pool

    def get_random_noise(self):
        """Ambil satu noise map acak dari pool."""
        idx = random.randint(0, len(self.noise_pool) - 1)
        return self.noise_pool[idx].copy()

    def __len__(self):
        return min(len(self.lr_frames), len(self.gt_frames))

    def __getitem__(self, idx):
        lr = self.lr_frames[idx]
        gt = self.gt_frames[idx]

        # Konversi ke tensor (C, H, W)
        lr_tensor = torch.from_numpy(lr).unsqueeze(0).float()  # (1, H_lr, W_lr)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()  # (1, H_hr, W_hr)

        # Ambil noise map
        noise = self.get_random_noise()
        noise_tensor = torch.from_numpy(noise).unsqueeze(0).float()  # (1, H_hr, W_hr)

        return lr_tensor, gt_tensor, noise_tensor


# ==================== Fungsi Utilitas ====================
def compute_psnr(output, target):
    """Hitung PSNR antara output dan target (range [0, 1])."""
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def export_weights_to_txt(model, output_dir='.'):
    """
    Export bobot model ke file .txt yang kompatibel dengan C code.
    Format: satu nilai per baris, urutan sesuai dengan cara C membaca.
    """
    print("\nMengekspor bobot ke file .txt...")

    layer_map = {
        'conv1': ('weights_layer1_finetuned.txt', 'biasess_layer1_finetuned.txt'),
        'conv2': ('weights_layer2_finetuned.txt', 'biasess_layer2_finetuned.txt'),
        'conv3': ('weights_layer3_finetuned.txt', 'biasess_layer3_finetuned.txt'),
        'conv4': ('weights_layer4_finetuned.txt', 'biasess_layer4_finetuned.txt'),
        'conv5': ('weights_layer5_finetuned.txt', 'biasess_layer5_finetuned.txt'),
        'conv6': ('weights_layer6_finetuned.txt', 'biasess_layer6_finetuned.txt'),
        'conv7': ('weights_layer7_finetuned.txt', 'biasess_layer7_finetuned.txt'),
        'deconv8': ('weights_layer8_finetuned.txt', 'biasess_layer8_finetuned.txt'),
    }

    for layer_name, (w_file, b_file) in layer_map.items():
        layer = getattr(model, layer_name)
        weight = layer.weight.data.cpu().numpy().flatten()
        bias = layer.bias.data.cpu().numpy().flatten()

        w_path = os.path.join(output_dir, w_file)
        b_path = os.path.join(output_dir, b_file)

        with open(w_path, 'w') as f:
            for val in weight:
                f.write(f'{val:.10f}\n')
        print(f"  ✅ {w_file} ({len(weight)} values)")

        with open(b_path, 'w') as f:
            for val in bias:
                f.write(f'{val:.10f}\n')
        print(f"  ✅ {b_file} ({len(bias)} values)")

    print("Export selesai!")


# ==================== Training ====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Inisialisasi model dan load bobot original
    model = FSRCNN(scale=CONFIG['scale']).to(device)

    if os.path.exists(CONFIG['weights_in']):
        model.load_state_dict(torch.load(CONFIG['weights_in'],
                                          map_location=device,
                                          weights_only=True))
        print(f"✅ Bobot original dimuat dari {CONFIG['weights_in']}")
    else:
        print(f"❌ File {CONFIG['weights_in']} tidak ditemukan!")
        print("   Jalankan txttopth.py terlebih dahulu.")
        return

    # 2. Siapkan dataset
    dataset = RaceConditionDataset(
        lr_video_path=CONFIG['lr_video'],
        data_root=CONFIG['data_root'],
        config=CONFIG
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 karena dataset kecil, hindari overhead multiprocessing
        drop_last=False
    )

    # 3. Optimizer dan Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.L1Loss()  # L1 lebih robust terhadap outlier dari race condition

    # 4. Evaluasi sebelum training (baseline)
    model.eval()
    with torch.no_grad():
        total_psnr_clean = 0
        total_psnr_noisy = 0
        count = 0
        for lr, gt, noise in dataloader:
            lr, gt, noise = lr.to(device), gt.to(device), noise.to(device)
            output_clean = model(lr)
            output_noisy = model(lr, noise_map=noise)
            for i in range(lr.size(0)):
                total_psnr_clean += compute_psnr(output_clean[i], gt[i])
                total_psnr_noisy += compute_psnr(output_noisy[i], gt[i])
                count += 1
        if count > 0:
            print(f"\n=== Baseline (sebelum fine-tuning) ===")
            print(f"  PSNR tanpa noise: {total_psnr_clean / count:.2f} dB")
            print(f"  PSNR dengan noise: {total_psnr_noisy / count:.2f} dB")
            print()

    # 5. Training loop
    print(f"Mulai fine-tuning: {CONFIG['num_epochs']} epoch, "
          f"lr={CONFIG['learning_rate']}, batch_size={CONFIG['batch_size']}")
    print(f"Noise probability: {CONFIG['noise_probability']}")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for lr, gt, noise in dataloader:
            lr, gt, noise = lr.to(device), gt.to(device), noise.to(device)

            optimizer.zero_grad()

            # Decide apakah inject noise (probabilistik)
            if random.random() < CONFIG['noise_probability']:
                output = model(lr, noise_map=noise)
            else:
                output = model(lr)

            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluasi PSNR setiap 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                total_psnr_clean = 0
                total_psnr_noisy = 0
                count = 0
                for lr_v, gt_v, noise_v in dataloader:
                    lr_v = lr_v.to(device)
                    gt_v = gt_v.to(device)
                    noise_v = noise_v.to(device)
                    out_c = model(lr_v)
                    out_n = model(lr_v, noise_map=noise_v)
                    for i in range(lr_v.size(0)):
                        total_psnr_clean += compute_psnr(out_c[i], gt_v[i])
                        total_psnr_noisy += compute_psnr(out_n[i], gt_v[i])
                        count += 1
                psnr_c = total_psnr_clean / max(count, 1)
                psnr_n = total_psnr_noisy / max(count, 1)
                print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                      f"Loss: {avg_loss:.6f}  "
                      f"PSNR(clean): {psnr_c:.2f} dB  "
                      f"PSNR(noisy): {psnr_n:.2f} dB")
        else:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                  f"Loss: {avg_loss:.6f}")

        # Simpan checkpoint
        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
            ckpt_path = f'fsrcnn_finetuned_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  💾 Checkpoint: {ckpt_path}")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG['weights_out'])

    print("=" * 60)
    print(f"Training selesai! Best loss: {best_loss:.6f}")
    print(f"Model terbaik disimpan di: {CONFIG['weights_out']}")

    # 6. Load best model dan evaluasi akhir
    model.load_state_dict(torch.load(CONFIG['weights_out'],
                                      map_location=device,
                                      weights_only=True))
    model.eval()
    with torch.no_grad():
        total_psnr_clean = 0
        total_psnr_noisy = 0
        count = 0
        for lr, gt, noise in dataloader:
            lr, gt, noise = lr.to(device), gt.to(device), noise.to(device)
            out_c = model(lr)
            out_n = model(lr, noise_map=noise)
            for i in range(lr.size(0)):
                total_psnr_clean += compute_psnr(out_c[i], gt[i])
                total_psnr_noisy += compute_psnr(out_n[i], gt[i])
                count += 1
        if count > 0:
            print(f"\n=== Evaluasi Akhir (setelah fine-tuning) ===")
            print(f"  PSNR tanpa noise: {total_psnr_clean / count:.2f} dB")
            print(f"  PSNR dengan noise: {total_psnr_noisy / count:.2f} dB")

    # 7. Export bobot ke .txt
    export_weights_to_txt(model, output_dir='.')

    print("\n🎉 Selesai! Langkah selanjutnya:")
    print("  1. Copy file *_finetuned.txt ke folder C code")
    print("  2. Rename menjadi weights_layerX.txt / biasess_layerX.txt")
    print("  3. Compile ulang C code dan jalankan tanpa #pragma omp critical")


# ==================== Entry Point ====================
if __name__ == '__main__':
    train()