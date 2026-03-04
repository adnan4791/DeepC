#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning FSRCNN v2: Channel Dropout Strategy

Perbedaan dari v1:
- Simulasi race condition = Channel Dropout di deconv Layer 8
  (bukan additive noise)
- Freeze layer 1-7, hanya train Layer 8
- Loss: MSE
- Curriculum dropout: rate naik bertahap
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
    'lr_width': 176,
    'lr_height': 144,
    'scale': 2,
    'num_frames': 150,
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-5,     # kecil agar tidak diverge
    'checkpoint_interval': 20,
    'dropout_start': 0.02,     # dropout rate awal (1 dari 56 channel)
    'dropout_end': 0.05,       # dropout rate akhir (2-3 dari 56 channel)
    'weight_reg': 0.01,        # L2 regularization terhadap bobot original
    'data_root': 'train_data',
    'lr_video': 'suzie_qcif.yuv',
    'weights_in': 'fsrcnn_original.pth',
    'weights_out': 'fsrcnn_finetuned_v2.pth',
}

# ==================== Model FSRCNN v2 ====================
class PReLU_custom(nn.Module):
    """PReLU dengan koefisien tetap (tidak trainable), sesuai C code."""
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.where(x > 0, x, self.coeff * x)


class FSRCNNv2(nn.Module):
    """
    FSRCNN dengan Channel Dropout di deconv Layer 8.
    
    Saat training: beberapa channel dari 56 input deconv di-drop secara acak,
    mensimulasikan race condition di mana imadd kehilangan kontribusi channel.
    
    Saat inference (eval mode): semua channel aktif, output identik dengan
    model original jika bobot tidak berubah.
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        # Layer 1-7 (akan di-freeze)
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
        # Layer 8: Deconvolution (TRAINABLE)
        self.deconv8 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2,
                                           padding=4, output_padding=1)
        
        # Channel dropout rate (diatur secara dinamis saat training)
        self.channel_dropout_rate = 0.1

    def forward(self, x):
        # Layer 1-7 (frozen)
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))
        x = self.prelu7(self.conv7(x))
        # x shape: (B, 56, H, W)
        
        if self.training and self.channel_dropout_rate > 0:
            # Channel Dropout: simulasi race condition
            # Buat mask per channel (B, 56, 1, 1)
            B, C, H, W = x.shape
            mask = torch.ones(B, C, 1, 1, device=x.device)
            # Untuk setiap sampel di batch, drop beberapa channel secara acak
            num_drop = max(1, int(C * self.channel_dropout_rate))
            for b in range(B):
                drop_indices = random.sample(range(C), num_drop)
                mask[b, drop_indices] = 0.0
            # Scale agar expected value tetap sama
            scale_factor = C / (C - num_drop)
            x = x * mask * scale_factor
        
        x = self.deconv8(x)
        return x

    def set_dropout_rate(self, rate):
        """Set channel dropout rate secara dinamis (untuk curriculum learning)."""
        self.channel_dropout_rate = rate


# Wrapper untuk kompatibilitas dengan state_dict dari model v1
class FSRCNN_compat(nn.Module):
    """Model original untuk load state_dict."""
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
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))
        x = self.prelu7(self.conv7(x))
        x = self.deconv8(x)
        return x


# ==================== Dataset ====================
class YUVDataset(Dataset):
    """Dataset: LR dari suzie_qcif.yuv, GT dari train_data/gt/."""
    def __init__(self, lr_video_path, gt_dir, config):
        self.config = config
        hr_h = config['lr_height'] * config['scale']
        hr_w = config['lr_width'] * config['scale']
        self.hr_shape = (hr_h, hr_w)
        self.lr_shape = (config['lr_height'], config['lr_width'])

        print("Memuat frame LR...")
        self.lr_frames = self._load_lr_frames(lr_video_path)
        print(f"  {len(self.lr_frames)} frame LR")

        print("Memuat frame GT...")
        self.gt_frames = self._load_gt_frames(gt_dir)
        print(f"  {len(self.gt_frames)} frame GT")

    def _load_lr_frames(self, path):
        frames = []
        h, w = self.lr_shape
        y_size = h * w
        uv_size = (h // 2) * (w // 2)
        frame_size = y_size + 2 * uv_size
        with open(path, 'rb') as f:
            for i in range(self.config['num_frames']):
                raw = f.read(frame_size)
                if len(raw) < frame_size:
                    break
                y = np.frombuffer(raw[:y_size], dtype=np.uint8)
                frames.append(y.reshape(h, w).astype(np.float32) / 255.0)
        return frames

    def _load_gt_frames(self, folder):
        frames = []
        h, w = self.hr_shape
        for i in range(self.config['num_frames']):
            path = os.path.join(folder, f'frame_{i:04d}.yuv')
            if not os.path.exists(path):
                break
            data = np.fromfile(path, dtype=np.uint8)
            if len(data) >= h * w:
                frames.append(data[:h*w].reshape(h, w).astype(np.float32) / 255.0)
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
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def export_weights_to_txt(model, output_dir='.', suffix='_finetuned_v2'):
    """Export bobot ke .txt kompatibel C code."""
    print(f"\nMengekspor bobot ke file *{suffix}.txt ...")
    layer_map = {
        'conv1': (f'weights_layer1{suffix}.txt', f'biasess_layer1{suffix}.txt'),
        'conv2': (f'weights_layer2{suffix}.txt', f'biasess_layer2{suffix}.txt'),
        'conv3': (f'weights_layer3{suffix}.txt', f'biasess_layer3{suffix}.txt'),
        'conv4': (f'weights_layer4{suffix}.txt', f'biasess_layer4{suffix}.txt'),
        'conv5': (f'weights_layer5{suffix}.txt', f'biasess_layer5{suffix}.txt'),
        'conv6': (f'weights_layer6{suffix}.txt', f'biasess_layer6{suffix}.txt'),
        'conv7': (f'weights_layer7{suffix}.txt', f'biasess_layer7{suffix}.txt'),
        'deconv8': (f'weights_layer8{suffix}.txt', f'biasess_layer8{suffix}.txt'),
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

        with open(b_path, 'w') as f:
            for val in bias:
                f.write(f'{val:.10f}\n')

        print(f"  ✅ {w_file} ({len(weight)}), {b_file} ({len(bias)})")

    print("Export selesai!")


# ==================== Training ====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load bobot original
    model = FSRCNNv2(scale=CONFIG['scale']).to(device)

    if os.path.exists(CONFIG['weights_in']):
        state_dict = torch.load(CONFIG['weights_in'], map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Bobot original dimuat dari {CONFIG['weights_in']}")
    else:
        print(f"❌ File {CONFIG['weights_in']} tidak ditemukan!")
        return

    # 2. FREEZE layer 1-7, hanya train Layer 8
    frozen_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7',
                     'prelu1', 'prelu2', 'prelu3', 'prelu4', 'prelu5', 'prelu6', 'prelu7']
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in frozen_layers:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params} trainable / {total_params} total")
    print(f"  (Hanya deconv8: weight {56*1*9*9}=4536 + bias 1 = 4537 params)")

    # 3. Dataset
    gt_dir = os.path.join(CONFIG['data_root'], 'gt')
    dataset = YUVDataset(CONFIG['lr_video'], gt_dir, CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True, num_workers=0)

    # 4. Optimizer & Loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()

    # Simpan bobot original Layer 8 untuk regularisasi
    original_deconv_weight = model.deconv8.weight.data.clone()
    original_deconv_bias = model.deconv8.bias.data.clone()

    # 5. Baseline evaluation
    model.eval()
    with torch.no_grad():
        total_psnr = 0
        count = 0
        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                total_psnr += compute_psnr(out[i], gt[i])
                count += 1
        print(f"\n=== Baseline PSNR (tanpa dropout): {total_psnr/count:.2f} dB ===\n")

    # 6. Training loop dengan curriculum dropout
    print(f"Training: {CONFIG['num_epochs']} epoch, lr={CONFIG['learning_rate']}")
    print(f"Dropout curriculum: {CONFIG['dropout_start']:.2f} → {CONFIG['dropout_end']:.2f}")
    print("=" * 60)

    best_psnr_with_dropout = 0

    for epoch in range(CONFIG['num_epochs']):
        # Curriculum: dropout rate naik bertahap
        progress = epoch / max(CONFIG['num_epochs'] - 1, 1)
        current_dropout = (CONFIG['dropout_start'] +
                           progress * (CONFIG['dropout_end'] - CONFIG['dropout_start']))
        model.set_dropout_rate(current_dropout)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, gt)
            # L2 regularization: jaga bobot dekat original
            reg_w = torch.mean((model.deconv8.weight - original_deconv_weight) ** 2)
            reg_b = torch.mean((model.deconv8.bias - original_deconv_bias) ** 2)
            loss = loss + CONFIG['weight_reg'] * (reg_w + reg_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluasi setiap 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                psnr_clean = 0
                psnr_drop = 0
                count = 0
                for lr_v, gt_v in dataloader:
                    lr_v, gt_v = lr_v.to(device), gt_v.to(device)
                    # Clean (tanpa dropout)
                    out_c = model(lr_v)
                    # Dengan dropout (simulasi race condition)
                    model.train()
                    model.set_dropout_rate(current_dropout)
                    out_d = model(lr_v)
                    model.eval()
                    for i in range(lr_v.size(0)):
                        psnr_clean += compute_psnr(out_c[i], gt_v[i])
                        psnr_drop += compute_psnr(out_d[i], gt_v[i])
                        count += 1
                pc = psnr_clean / max(count, 1)
                pd = psnr_drop / max(count, 1)
                print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                      f"Loss: {avg_loss:.8f}  "
                      f"Drop: {current_dropout:.3f}  "
                      f"PSNR(clean): {pc:.2f} dB  "
                      f"PSNR(drop): {pd:.2f} dB")

                # Track best (berdasarkan PSNR dengan dropout)
                if pd > best_psnr_with_dropout:
                    best_psnr_with_dropout = pd
                    torch.save(model.state_dict(), CONFIG['weights_out'])
        else:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}]  "
                  f"Loss: {avg_loss:.8f}  "
                  f"Drop: {current_dropout:.3f}")

        # Checkpoint
        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
            ckpt = f'fsrcnn_finetuned_v2_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 Checkpoint: {ckpt}")

    print("=" * 60)
    print(f"Training selesai! Best PSNR(dropout): {best_psnr_with_dropout:.2f} dB")

    # 7. Load best & final evaluation
    model.load_state_dict(torch.load(CONFIG['weights_out'], map_location=device,
                                      weights_only=True))
    model.eval()
    with torch.no_grad():
        total_psnr = 0
        count = 0
        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            out = model(lr)
            for i in range(lr.size(0)):
                total_psnr += compute_psnr(out[i], gt[i])
                count += 1
        print(f"\n=== Final PSNR (tanpa dropout): {total_psnr/count:.2f} dB ===")

    # 8. Export
    export_weights_to_txt(model, output_dir='.', suffix='_finetuned_v2')

    print("\n🎉 Selesai!")
    print("  File bobot: *_finetuned_v2.txt")
    print("  Untuk penggunaan di C: ganti weights_layer8.txt & biasess_layer8.txt")
    print("  (Layer 1-7 TIDAK BERUBAH, cukup ganti layer 8)")


if __name__ == '__main__':
    train()