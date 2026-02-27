import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ctypes
import os

# ==========================================
# 1. LOAD MODUL C (SHARED OBJECT)
# ==========================================
# Pastikan nama file .so sesuai dengan yang Anda kompilasi
lib = ctypes.CDLL(os.path.abspath("./fsrcnn_race_sim.so"))

# Deklarasi tipe argumen untuk fungsi C
# void simulate_layer8_race(double *in_56, double *out_1, int out_rows, int out_cols, int num_channels)
lib.simulate_layer8_race.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# ==========================================
# 2. DEFINISI AUTOGRAD CUSTOM PYTORCH
# ==========================================
class RaceConditionImadd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # input_tensor: (Batch, 56, H, W)
        ctx.save_for_backward(input_tensor)
        
        batch_size, num_channels, height, width = input_tensor.shape
        output_tensor = torch.zeros((batch_size, 1, height, width), dtype=torch.float64)
        
        # Eksekusi fungsi C untuk setiap gambar dalam batch
        for b in range(batch_size):
            in_np = input_tensor[b].detach().cpu().numpy().astype(np.float64).flatten()
            out_np = np.zeros(height * width, dtype=np.float64)
            
            # Panggil C wrapper (Race condition terjadi di sini!)
            lib.simulate_layer8_race(in_np, out_np, height, width, num_channels)
            
            output_tensor[b, 0, :, :] = torch.from_numpy(out_np.reshape(height, width))
            
        return output_tensor.to(input_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: Error gradient dari loss function (Batch, 1, H, W)
        input_tensor, = ctx.saved_tensors
        
        # TRIK MATEMATIKA: Karena operasi ke depannya adalah "Sum" (meskipun cacat),
        # cara terbaik untuk backpropagate adalah dengan mendistribusikan error 
        # secara merata ke seluruh 56 channel.
        grad_input = grad_output.expand_as(input_tensor) 
        
        return grad_input

# ==========================================
# 3. DEFINISI MODEL LAYER 8
# ==========================================
class Layer8FSRCNN(nn.Module):
    def __init__(self, scale):
        super(Layer8FSRCNN, self).__init__()
        # Di C, Deconv Layer 8 memproses 56 channel secara independen.
        # Ini disebut "Depthwise Convolution" di PyTorch (groups=56).
        # Kernel 9x9, stride=scale. Padding disesuaikan agar output pas rows*scale x cols*scale.
        padding = 9 // 2
        output_padding = scale - 1
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=56, out_channels=56, 
            kernel_size=9, stride=scale, 
            padding=padding, output_padding=output_padding, 
            groups=56, bias=False, dtype=torch.float64
        )
        # Bias akhir untuk 1 channel (skalar)
        self.bias = nn.Parameter(torch.tensor([-0.03262640000], dtype=torch.float64))
        
        # Opsional: Load bobot awal dari file lama agar training lebih cepat hội tụ
        self.load_initial_weights()

    def load_initial_weights(self):
        if os.path.exists("weights_layer8_original.txt"):
            w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64)
            # Reshape ke format PyTorch: (in_channels, out_channels/groups, kH, kW)
            w = w.reshape(56, 1, 9, 9)
            self.deconv.weight.data = torch.from_numpy(w)
            print("Berhasil memuat weights_layer8_original.txt sebagai titik awal.")

    def forward(self, x):
        # 1. Upscale 56 channel (Aman)
        x = self.deconv(x)
        # 2. Imadd dengan Race Condition (Cacat Hardware)
        x = RaceConditionImadd.apply(x)
        # 3. Tambah Bias
        x = x + self.bias
        return x

# ==========================================
# 4. DATALOADER
# ==========================================
class FSRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        # Baca dimensi dari file yang dibuat oleh program C
        with open(os.path.join(data_dir, "dimensions.txt"), "r") as f:
            self.in_rows, self.in_cols, self.scale, self.num_frames = map(int, f.read().split())
            
        self.out_rows = self.in_rows * self.scale
        self.out_cols = self.in_cols * self.scale

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # Load Layer 7 (Input ke Layer 8)
        l7_path = os.path.join(self.data_dir, f"layer7_frame{idx:04d}.bin")
        l7_data = np.fromfile(l7_path, dtype=np.float64).reshape(56, self.in_rows, self.in_cols)
        
        # Load HR Target (Ground Truth)
        hr_path = os.path.join(self.data_dir, f"hr_frame{idx:04d}.bin")
        hr_data = np.fromfile(hr_path, dtype=np.float64).reshape(1, self.out_rows, self.out_cols)
        
        return torch.from_numpy(l7_data), torch.from_numpy(hr_data)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    dataset = FSRCNNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = Layer8FSRCNN(scale=dataset.scale)
    criterion = nn.L1Loss() # L1 Loss (MAE) biasanya lebih baik untuk super resolusi
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 20 # Anda bisa menaikkan ini nanti
    print("\nMemulai proses adaptasi model (Training)...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (layer7_input, hr_target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass: PyTorch -> C (Race Condition) -> PyTorch
            output = model(layer7_input)
            
            # Hitung error terhadap gambar HR yang sempurna
            loss = criterion(output, hr_target)
            
            # Backward pass & Update bobot!
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.6f}")

    print("\nTraining selesai! Mengekspor bobot baru...")
    
    # ==========================================
    # 6. EKSPOR BOBOT KEMBALI KE C
    # ==========================================
    # Ambil bobot deconv dan flatten ke 1D array
    new_weights = model.deconv.weight.data.numpy().flatten()
    new_bias = model.bias.data.item()
    
    # Simpan ke txt
    np.savetxt("weights_layer8_trained.txt", new_weights, fmt="%.10f")
    
    with open("biasess_layer8_trained.txt", "w") as f:
        f.write(f"{new_bias:.10f}\n")
        
    print("Bobot baru tersimpan di 'weights_layer8_trained.txt' dan 'biasess_layer8_trained.txt'.")
    print("Silakan ganti/rename file ini di program C Anda dan jalankan inferensi!")

if __name__ == "__main__":
    train()