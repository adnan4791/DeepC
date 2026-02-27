import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class Layer8FSRCNN_Sparse(nn.Module):
    def __init__(self, scale):
        super(Layer8FSRCNN_Sparse, self).__init__()
        padding = 9 // 2
        output_padding = scale - 1
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=56, out_channels=56, 
            kernel_size=9, stride=scale, 
            padding=padding, output_padding=output_padding, 
            groups=56, bias=False, dtype=torch.float64
        )
        self.bias = nn.Parameter(torch.tensor([-0.03262640000], dtype=torch.float64))
        
        if os.path.exists("weights_layer8.txt"): # Ganti dengan nama bobot original Anda
            w = np.loadtxt("weights_layer8.txt", dtype=np.float64).reshape(56, 1, 9, 9)
            self.deconv.weight.data = torch.from_numpy(w)

    def forward(self, x):
        # 1. Upscale 56 channel
        channels_out = self.deconv(x)
        # 2. Jumlahkan secara normal (aman di PyTorch)
        out_sum = torch.sum(channels_out, dim=1, keepdim=True) + self.bias
        # Kembalikan hasil sum dan raw channels (untuk dihitung penaltinya)
        return out_sum, channels_out

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
    
    model = Layer8FSRCNN_Sparse(scale=dataset.scale)
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # INI ADALAH KUNCI UTAMANYA: Seberapa kuat kita memaksa channel menjadi 0
    lambda_sparsity = 0.05 
    epochs = 20
    
    print("\nMemulai training dengan strategi Sparsity (Anti-Race Condition)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for layer7_input, hr_target in dataloader:
            optimizer.zero_grad()
            
            output_sum, channels_out = model(layer7_input)
            
            # 1. Loss utama: Gambar harus mirip dengan Ground Truth
            loss_reconstruction = criterion_mae(output_sum, hr_target)
            
            # 2. Loss Sparsity (Penalti): Memaksa ke-56 channel nilainya sedekat mungkin ke 0
            loss_sparsity = lambda_sparsity * torch.mean(torch.abs(channels_out))
            
            # Total loss
            loss = loss_reconstruction + loss_sparsity
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {epoch_loss/len(dataloader):.6f}")

    print("\nMengekspor bobot Sparsity...")
    np.savetxt("weights_layer8_sparse.txt", model.deconv.weight.data.numpy().flatten(), fmt="%.10f")
    with open("biasess_layer8_sparse.txt", "w") as f: f.write(f"{model.bias.data.item():.10f}\n")
    print("Selesai! Silakan uji bobot '_sparse.txt' ini.")

if __name__ == "__main__":
    train()