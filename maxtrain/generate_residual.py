import torch
import torch.nn as nn
import numpy as np
import os

class RaceCompensator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


race_dir = "dataset/race"
out_dir = "dataset/compensated_residual"

os.makedirs(out_dir, exist_ok=True)

# load model
model = RaceCompensator()
model.load_state_dict(torch.load("race_residual_compensator.pth", map_location="cpu"))
model.eval()


for i in range(150):

    race = np.load(f"{race_dir}/frame_{i:03d}_run1.npy")

    # normalisasi sama seperti training
    x = torch.from_numpy(race).float()/255.0
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        residual = model(x)

    # kembali ke numpy
    residual = residual.squeeze().numpy()

    race_norm = race / 255.0

    corrected = race_norm + residual

    # kembali ke skala 0-255
    corrected = corrected * 255.0

    corrected = np.clip(corrected, 0, 255)
    corrected = corrected.astype(np.uint8)

    np.save(f"{out_dir}/frame_{i:03d}.npy", corrected)

print("DONE generating compensated frames")