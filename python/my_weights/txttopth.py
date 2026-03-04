import torch
import torch.nn as nn
import numpy as np
import os

# ==================== Definisi Model ====================
class PReLU_custom(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        return torch.where(x > 0, x, self.coeff * x)

class FSRCNN(nn.Module):
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
        # Perhatikan: ConvTranspose2d(in_channels, out_channels, ...)
        self.deconv8 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2, padding=4, output_padding=1)

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

# ==================== Fungsi Pembaca ====================
def load_weights_from_txt(filepath, shape):
    """Membaca file teks berisi float (bisa multiple per baris), mengabaikan baris kosong dan komentar.
       Jika jumlah angka lebih besar dari yang dibutuhkan, hanya diambil sejumlah yang diperlukan.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} tidak ditemukan")
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Pisahkan berdasarkan whitespace
            parts = line.split()
            for part in parts:
                try:
                    val = float(part)
                    data.append(val)
                except ValueError:
                    print(f"Peringatan: tidak dapat mengurai '{part}' di file {filepath}")
    
    total_needed = np.prod(shape)
    if len(data) < total_needed:
        raise ValueError(f"File {filepath} hanya memiliki {len(data)} angka, butuh {total_needed}")
    elif len(data) > total_needed:
        print(f"File {filepath} memiliki {len(data)} angka, tetapi hanya {total_needed} yang diperlukan. Mengambil {total_needed} pertama.")
        data = data[:total_needed]
    
    tensor = torch.tensor(data).view(shape)
    return tensor

# ==================== Main ====================
def main():
    # Cek keberadaan file
    files = [
        'weights_layer1.txt', 'biasess_layer1.txt',
        'weights_layer2.txt', 'biasess_layer2.txt',
        'weights_layer3.txt', 'biasess_layer3.txt',
        'weights_layer4.txt', 'biasess_layer4.txt',
        'weights_layer5.txt', 'biasess_layer5.txt',
        'weights_layer6.txt', 'biasess_layer6.txt',
        'weights_layer7.txt', 'biasess_layer7.txt',
        'weights_layer8.txt', 'biasess_layer8.txt'
    ]
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f} ditemukan")
        else:
            print(f"❌ {f} TIDAK ditemukan")
            return

    model = FSRCNN(scale=2)

    # Layer 1
    model.conv1.weight.data = load_weights_from_txt('weights_layer1.txt', (56, 1, 5, 5))
    model.conv1.bias.data = load_weights_from_txt('biasess_layer1.txt', (56,))

    # Layer 2
    model.conv2.weight.data = load_weights_from_txt('weights_layer2.txt', (12, 56, 1, 1))
    model.conv2.bias.data = load_weights_from_txt('biasess_layer2.txt', (12,))

    # Layer 3
    model.conv3.weight.data = load_weights_from_txt('weights_layer3.txt', (12, 12, 3, 3))
    model.conv3.bias.data = load_weights_from_txt('biasess_layer3.txt', (12,))

    # Layer 4
    model.conv4.weight.data = load_weights_from_txt('weights_layer4.txt', (12, 12, 3, 3))
    model.conv4.bias.data = load_weights_from_txt('biasess_layer4.txt', (12,))

    # Layer 5
    model.conv5.weight.data = load_weights_from_txt('weights_layer5.txt', (12, 12, 3, 3))
    model.conv5.bias.data = load_weights_from_txt('biasess_layer5.txt', (12,))

    # Layer 6
    model.conv6.weight.data = load_weights_from_txt('weights_layer6.txt', (12, 12, 3, 3))
    model.conv6.bias.data = load_weights_from_txt('biasess_layer6.txt', (12,))

    # Layer 7
    model.conv7.weight.data = load_weights_from_txt('weights_layer7.txt', (56, 12, 1, 1))
    model.conv7.bias.data = load_weights_from_txt('biasess_layer7.txt', (56,))

    # Layer 8 (deconv) - PERBAIKAN: shape (56, 1, 9, 9)
    model.deconv8.weight.data = load_weights_from_txt('weights_layer8.txt', (56, 1, 9, 9))
    model.deconv8.bias.data = load_weights_from_txt('biasess_layer8.txt', (1,))

    print("Semua bobot berhasil dimuat!")

    # Simpan model
    torch.save(model.state_dict(), 'fsrcnn_original.pth')
    print("Model disimpan sebagai fsrcnn_original.pth")

if __name__ == '__main__':
    main()