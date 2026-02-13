
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os

# Configuration
LAYER7_FILE = 'layer7.bin'
TARGET_FILE = 'akiyo_cif.yuv' # Use the CIF YUV as target
NUM_FRAMES = 50
CHANNELS = 56
ROWS = 144
COLS = 176
SCALE = 2
FILTERS_8 = 1
FILTER_SIZE_8 = 9
DROPOUT_RATE = 0.2 # Simulate race condition loss

class Layer8Dataset(Dataset):
    def __init__(self, layer7_path, target_path, num_frames, rows, cols, channels):
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_frames = num_frames
        
        # Load Layer 7 inputs (Features)
        # Format: double (8 bytes)
        # Size: num_frames * (rows * cols * channels)
        # We read all into memory (approx 50 * 144 * 176 * 56 * 8 bytes ≈ 500MB). OK.
        print("Loading input features...")
        with open(layer7_path, 'rb') as f:
            data = f.read()
            # shape (num_frames, channels, rows, cols) ? 
            # C code: imadd(img_fltr_p7 + i*rows*cols, ...)
            # i is channel index (0..55).
            # So memory layout is likely: Frame -> Channel -> Row -> Col ?
            # Wait, looking at source:
            # allocate `img_fltr_7` size `rows*cols*56`.
            # Access: `img_fltr_7 + j*rows*cols`. j is channel.
            # So consecutive memory is: Channel 0, Channel 1...
            # This matches PyTorch (C, H, W).
            self.inputs = np.frombuffer(data, dtype=np.float64).reshape(num_frames, channels, rows, cols)
            self.inputs = torch.from_numpy(self.inputs).float()
            
        # Load Target (HR Y-channel)
        print("Loading targets...")
        self.targets = []
        width_hr = cols * SCALE
        height_hr = rows * SCALE
        y_size = width_hr * height_hr
        uv_size = (width_hr // 2) * (height_hr // 2)
        frame_size = y_size + 2 * uv_size
        
        with open(target_path, 'rb') as f:
            for i in range(num_frames):
                # Seek to frame
                f.seek(i * frame_size)
                y_data = f.read(y_size)
                # Y data is uint8.
                y_img = np.frombuffer(y_data, dtype=np.uint8).reshape(height_hr, width_hr)
                # Normalize to 0-1? source.c outputs in 0-1 and then scales to 255.
                # source.c loads uint8 input, divides by 255.0.
                # So target should be 0-1?
                # Actually source.c Layer 8 output is added to `biases_layer8`.
                # And finally converted to uint8.
                # We should train against 0-255 or 0-1?
                # Inputs (Layer 7) are likely valid range.
                # Let's normalize target to 0-1 to match network scale likely.
                self.targets.append(y_img.astype(np.float32) / 255.0)
        
        self.targets = np.stack(self.targets)
        # Add channel dim: (N, 1, H, W)
        self.targets = torch.from_numpy(self.targets).unsqueeze(1)
        
        # Verify sizes
        print(f"Inputs: {self.inputs.shape}")
        print(f"Targets: {self.targets.shape}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class RobustDeconvLayer(nn.Module):
    def __init__(self):
        super(RobustDeconvLayer, self).__init__()
        # We need to simulate summing 56 separate deconvs.
        # Standard ConvTranspose2d(56, 1, ...) sums internally.
        # To inject noise *during* summation, we act as if we have 56 separate filters.
        # We can use groups=1 (standard) and do trickery, OR use groups=56?
        # If groups=56, input 56 channels -> output 56 channels.
        # Each channel processed independently.
        # Then we Sum(outputs).
        # This allows us to apply Dropout to the 56 outputs before summing.
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=56, 
            out_channels=56, 
            kernel_size=9, 
            stride=2, 
            padding=4, 
            output_padding=1,
            groups=56, # Independent channels
            bias=False # Bias is applied after sum
        )
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Dropout
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x):
        # x: (N, 56, 144, 176)
        # out: (N, 56, 288, 352)
        out = self.deconv(x)
        
        # Apply dropout to the channels (simulate lost race condition updates)
        # We want to drop entire channels? Or random pixels?
        # User said "race condition saat menjumlahkan 56 channel".
        # If using standard dropout, it drops elements independently.
        # This simulates fine-grained race condition (pixel level).
        out = self.dropout(out)
        
        # Sum across channels to get single Y channel
        # out: (N, 1, 288, 352)
        out = torch.sum(out, dim=1, keepdim=True)
        
        # Add bias
        out = out + self.bias
        return out

def save_weights(model, path):
    # Extract weights and save in format expected by source.c
    # source.c expects weights_layer8.txt
    # Loop i=0..num_filters8 (1)
    #   Loop j=0..num_channels8 (56)
    #     Loop kernel (9x9)
    # But we used groups=56.
    # self.deconv.weight shape: (56, 1, 9, 9) because (in_channels, out_channels/groups, k, k)
    # We essentially have 56 filters, each taking 1 input.
    # We need to map this to (1 filter, 56 channels).
    # It is effectively the same values.
    # channel j weight in PyTorch `deconv.weight[j, 0, :, :]` corresponds to `kernel` for channel j in source.c
    
    weights = model.deconv.weight.data.cpu().numpy() # (56, 1, 9, 9)
    bias = model.bias.data.cpu().numpy() # (1)
    
    print(f"Saving weights to {path}...")
    with open(path, 'w') as f:
        # source.c order:
        # Channel 0, Channel 1, ... Channel 55
        for j in range(56):
            k = weights[j, 0, :, :] # 9x9
            k_flat = k.flatten()
            for val in k_flat:
                f.write(f"{val:.10f}\n")
                
    print(f"Learned Bias: {bias[0]:.10f}")
    with open("bias_layer8_new.txt", "w") as f:
        f.write(f"{bias[0]:.10f}")

def main():
    if not os.path.exists(LAYER7_FILE):
        print(f"Error: {LAYER7_FILE} not found.")
        return
        
    dataset = Layer8Dataset(LAYER7_FILE, TARGET_FILE, NUM_FRAMES, ROWS, COLS, CHANNELS)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = RobustDeconvLayer()
    
    # Initialize weights? Or load existing weights to start with?
    # Better to start from scratch or existing?
    # If we load existing, we fine tune.
    # Existing weights are in `weights/weights_layer8.txt`.
    # Let's try to load them if possible, but parsing them is work.
    # Random init is fine for this task usually.
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training...")
    for epoch in range(10): # Quick training
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.6f}")
        
    # Save
    save_weights(model, 'weights_layer8_robust.txt')
    print("Weights saved.")

if __name__ == "__main__":
    main()
