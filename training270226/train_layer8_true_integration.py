import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ctypes
import os

# ==========================================
# 1. LOAD MODUL C (SHARED OBJECT)
# ==========================================
lib = ctypes.CDLL(os.path.abspath("./fsrcnn_race_sim.so"))

lib.simulate_layer8_race.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# ==========================================
# 2. DEFINISI AUTOGRAD CUSTOM (STEP 2 & 3)
# ==========================================
class RaceConditionStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_56ch, bias):
        """
        input_56ch: (B, 56, H, W) -> Hasil Step 1 (Deconv)
        bias: Skalar Parameter    -> Step 3
        """
        batch_size, num_channels, height, width = input_56ch.shape
        output_tensor = torch.zeros((batch_size, 1, height, width), dtype=torch.float64)
        
        # Simpan tensor untuk backward
        ctx.save_for_backward(input_56ch)

        for b in range(batch_size):
            in_np = input_56ch[b].detach().cpu().numpy().astype(np.float64).flatten()
            out_np = np.zeros(height * width, dtype=np.float64)
            
            # --- STEP 2: Imadd dengan Race Condition di C ---
            lib.simulate_layer8_race(in_np, out_np, height, width, num_channels)
            
            # --- STEP 3: Tambah Bias (Dilakukan di sini agar terikat proses) ---
            # Kita lakukan di level numpy agar mereplikasi behavior hardware sepenuhnya
            out_np = out_np + bias.item()
            
            output_tensor[b, 0, :, :] = torch.from_numpy(out_np.reshape(height, width))
            
        return output_tensor.to(input_56ch.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: Error dari Loss (B, 1, H, W)
        """
        # Gradien untuk Step 2: Distribusi error ke 56 channel
        grad_input = grad_output.expand(-1, 56, -1, -1)
        
        # Gradien untuk Step 3 (Bias): Akumulasi seluruh error piksel
        # Karena (x + bias), maka turunan terhadap bias adalah 1.
        grad_bias = grad_output.sum() 
        
        return grad_input, grad_bias

# ==========================================
# 3. DEFINISI MODEL LAYER 8 (TRUE HYBRID)
# ==========================================
class Layer8FSRCNN(nn.Module):
    def __init__(self, scale):
        super(Layer8FSRCNN, self).__init__()
        # STEP 1: Deconvolution (Bersih di PyTorch)
        padding = 9 // 2
        output_padding = scale - 1
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=56, out_channels=56, 
            kernel_size=9, stride=scale, 
            padding=padding, output_padding=output_padding, 
            groups=56, bias=False, dtype=torch.float64
        )
        
        # PARAMETER BIAS: Akan ikut belajar di backward
        self.bias = nn.Parameter(torch.tensor([-0.0326264], dtype=torch.float64))
        
        self.load_initial_weights()

    def load_initial_weights(self):
        if os.path.exists("weights_layer8_original.txt"):
            w = np.loadtxt("weights_layer8_original.txt", dtype=np.float64).reshape(56, 1, 9, 9)
            self.deconv.weight.data = torch.from_numpy(w)
            print("Berhasil memuat weights_layer8_original.txt")

    def forward(self, x):
        # Step 1: Deconv
        x = self.deconv(x)
        # Step 2 & 3: Race Condition + Bias dalam satu fungsi Autograd
        x = RaceConditionStep.apply(x, self.bias)
        return x

# ==========================================
# 4. DATALOADER
# ==========================================
class FSRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "dimensions.txt"), "r") as f:
            self.in_rows, self.in_cols, self.scale, self.num_frames = map(int, f.read().split())
            
    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        l7_path = os.path.join(self.data_dir, f"layer7_frame{idx:04d}.bin")
        l7_data = np.fromfile(l7_path, dtype=np.float64).reshape(56, self.in_rows, self.in_cols)
        hr_path = os.path.join(self.data_dir, f"hr_frame{idx:04d}.bin")
        hr_data = np.fromfile(hr_path, dtype=np.float64).reshape(1, self.in_rows * self.scale, self.in_cols * self.scale)
        return torch.from_numpy(l7_data), torch.from_numpy(hr_data)

# ==========================================
# 5. TRAINING LOOP (AGGRESSIVE PSNR MODE)
# ==========================================
def train():
    dataset = FSRCNNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = Layer8FSRCNN(scale=dataset.scale)
    criterion = nn.MSELoss() # Langsung incar PSNR tinggi
    
    # Gunakan LR sedikit lebih tinggi di awal karena kita ingin bias bergerak
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 100 # Naikkan epoch untuk 150 frame agar konvergen
    print(f"\nMemulai adaptasi 3-Steps (Frames: {dataset.num_frames})...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (layer7_input, hr_target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            output = model(layer7_input)
            loss = criterion(output, hr_target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.10f} | Bias: {model.bias.item():.6f}")

    # EXPORT
    np.savetxt("weights_layer8_trained.txt", model.deconv.weight.data.numpy().flatten(), fmt="%.17e", newline="  ")
    with open("biasess_layer8_trained.txt", "w") as f:
        f.write(f"{model.bias.item():.10f}\n")
    print("\nSelesai! Bias dan Weights telah diupdate berdasarkan simulasi 3-langkah.")

if __name__ == "__main__":
    train()