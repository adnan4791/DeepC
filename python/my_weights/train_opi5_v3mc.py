import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess
import time

# ==========================================================
# CONFIGURATION FOR ORANGE PI 5
# ==========================================================
EXEC_V3 = "./fsrcnn_parallel_layer8v3"  # Base model (v3)
INPUT_YUV = "suzie_qcif.yuv"            # Input low-res
GT_YUV = "clean.yuv"                    # Ground Truth (HD Source)
SCALE = 2
WIDTH, HEIGHT = 176, 144                # Input resolution
OUT_WIDTH, OUT_HEIGHT = WIDTH*SCALE, HEIGHT*SCALE
NUM_FRAMES = 150                        # Dataset frames
THREADS = 8                             # Thread count to capture race conditions
GEN_RUNS = 10                           # Capture 10 noisy runs
TRAIN_EPOCHS = 100
BATCH_SIZE = 16

# ==========================================================
# MICRO-COMPENSATOR (8 Features, 2 Layers)
# ==========================================================
class MicroCompensator(nn.Module):
    def __init__(self):
        super(MicroCompensator, self).__init__()
        # Layer 1: Conv 1->8, 3x3, ReLU
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Layer 2: Conv 8->1, 3x3
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def load_yuv_frame(filename, frame_idx, w, h):
    frame_size = w * h + (w // 2) * (h // 2) * 2
    with open(filename, "rb") as f:
        f.seek(frame_idx * frame_size)
        data = f.read(w * h)
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0

# ==========================================================
# DATA GENERATION ON ORANGE PI 5
# ==========================================================
def generate_opi5_data():
    if not os.path.exists(EXEC_V3):
        print(f"Error: {EXEC_V3} not found. Please compile it first.")
        return None, None

    print(f"--- Step 1: Generating local noisy frames on Orange Pi 5 ({THREADS} threads) ---")
    os.makedirs("opi5_train", exist_ok=True)
    
    noisy_samples = []
    gt_samples = []

    # Load GT frames (HD Source)
    print(f"Loading Ground Truth from {GT_YUV}...")
    gt_frames = [load_yuv_frame(GT_YUV, i, OUT_WIDTH, OUT_HEIGHT) for i in range(NUM_FRAMES)]

    # Run V3 model N times to capture local race conditions
    for r in range(GEN_RUNS):
        tmp_out = f"opi5_train/run_{r}.yuv"
        print(f"  Run {r+1}/{GEN_RUNS} using {THREADS} threads...")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(THREADS)
        subprocess.run([EXEC_V3, INPUT_YUV, tmp_out], env=env, stdout=subprocess.DEVNULL)
        
        # Load output
        for i in range(NUM_FRAMES):
            noisy = load_yuv_frame(tmp_out, i, OUT_WIDTH, OUT_HEIGHT)
            noisy_samples.append(noisy)
            gt_samples.append(gt_frames[i])
        
        # Cleanup to save space
        os.remove(tmp_out)

    return np.array(noisy_samples), np.array(gt_samples)

# ==========================================================
# TRAINING
# ==========================================================
def train():
    device = torch.device("cpu") # OPi5 CPU is enough
    print(f"Using Device: {device}")

    # Gen / Load data
    inputs, targets = generate_opi5_data()
    if inputs is None: return

    # Convert to Tensors
    inputs = torch.from_numpy(inputs).unsqueeze(1)
    targets = torch.from_numpy(targets).unsqueeze(1)

    model = MicroCompensator().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"--- Step 2: Training Micro-Compensator ({len(inputs)} samples) ---")
    for epoch in range(TRAIN_EPOCHS):
        # Full batch or split if memory is tight, but 1.5k frames is fine
        indices = torch.randperm(len(inputs))
        total_loss = 0
        
        for i in range(0, len(inputs), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            x = inputs[batch_idx].to(device)
            y = targets[batch_idx].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{TRAIN_EPOCHS}] Loss: {total_loss/(len(inputs)/BATCH_SIZE):.6f}")

    # ==================== EXPORT WEIGHTS ====================
    print("--- Step 3: Exporting OPi5 Optimized Weights ---")
    with open("micro_compensator_weights.txt", "w") as f:
        f.write("# OPi5 Optimized Weights (v3 base)\n")
        # conv1.weight: (8, 1, 3, 3)
        for val in model.conv1.weight.detach().numpy().flatten():
            f.write(f"{val:.10f}\n")
        # conv1.bias: (8,)
        for val in model.conv1.bias.detach().numpy().flatten():
            f.write(f"{val:.10f}\n")
        # conv2.weight: (1, 8, 3, 3)
        for val in model.conv2.weight.detach().numpy().flatten():
            f.write(f"{val:.10f}\n")
        # conv2.bias: (1,)
        for val in model.conv2.bias.detach().numpy().flatten():
            f.write(f"{val:.10f}\n")

    print("Export Done! Use 'micro_compensator_weights.txt' with your C code.")

if __name__ == "__main__":
    train()
