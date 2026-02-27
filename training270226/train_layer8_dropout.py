import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class Layer8DropoutFSRCNN(nn.Module):
    def __init__(self, scale, drop_prob=0.3):
        super(Layer8DropoutFSRCNN, self).__init__()
        padding = 9 // 2
        output_padding = scale - 1
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=56, out_channels=56, 
            kernel_size=9, stride=scale, 
            padding=padding, output_padding=output_padding, 
            groups=56, bias=False, dtype=torch.float64
        )
        self.bias = nn.Parameter(torch.tensor([-0.03262640000], dtype=torch.float64))
        self.drop_prob = drop_prob
        
        # Load bobot original sebagai awalan
        if os.path.exists("weights_layer8_original.txt"):
            w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64).reshape(56, 1, 9, 9)
            self.deconv.weight.data = torch.from_numpy(w)
            print("Berhasil memuat weights_layer8_original.txt")

    def forward(self, x):
        # 1. Upscale 56 channel
        x = self.deconv(x)
        
        # 2. SIMULASI RACE CONDITION (Hanya aktif saat training)
        if self.training:
            # Buat mask: 1 (selamat), 0 (hilang kena race condition). 
            # probabilitas hilang = drop_prob (misal 30%)
            mask = torch.bernoulli(torch.full((x.shape[0], 56, 1, 1), 1 - self.drop_prob)).to(x.device)
            # Karena mask membuat sebagian channel jadi 0, kita tidak melakukan scaling ulang 
            # agar model benar-benar merasakan "kehilangan" energi sinyal (mirip cacat aslinya)
            x = x * mask 

        # 3. Jumlahkan ke 1 channel secara aman (meniru imadd)
        out = torch.sum(x, dim=1, keepdim=True) + self.bias
        return out

class FSRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "dimensions.txt"), "r") as f:
            self.in_rows, self.in_cols, self.scale, self.num_frames = map(int, f.read().split())
        self.out_rows, self.out_cols = self.in_rows * self.scale, self.in_cols * self.scale

    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        l7 = np.fromfile(os.path.join(self.data_dir, f"layer7_frame{idx:04d}.bin"), dtype=np.float64).reshape(56, self.in_rows, self.in_cols)
        hr = np.fromfile(os.path.join(self.data_dir, f"hr_frame{idx:04d}.bin"), dtype=np.float64).reshape(1, self.out_rows, self.out_cols)
        return torch.from_numpy(l7), torch.from_numpy(hr)

def train():
    dataset = FSRCNNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Kita set Drop Probability = 0.3 (menganggap sekitar 30% operasi imadd gagal/tertimpa)
    model = Layer8DropoutFSRCNN(scale=dataset.scale, drop_prob=0.3)
    model.train() # Pastikan mode training agar Dropout aktif
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 20
    
    print("\nMemulai training dengan Simulasi Dropout (V2)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for layer7_input, hr_target in dataloader:
            optimizer.zero_grad()
            output = model(layer7_input)
            loss = criterion(output, hr_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.6f}")

    print("\nTraining selesai! Mengekspor bobot Dropout...")
    # Paksa mode evaluasi sebelum ekspor agar aman
    model.eval() 
    np.savetxt(
        "weights_layer8_trained.txt",  # atau weights_layer8_dropout.txt
        model.deconv.weight.data.numpy().flatten(), 
        fmt="%.17e",    # Gunakan scientific notation (e) dengan presisi tinggi
        newline=" "     # Gunakan Spasi sebagai pemisah antar angka, BUKAN Enter!
    )
    with open("biasess_layer8_dropout.txt", "w") as f: 
        f.write(f"{model.bias.data.item():.10f}\n")
    print("Selesai! File bobot 'weights_layer8_dropout.txt' siap diuji.")

if __name__ == "__main__":
    train()