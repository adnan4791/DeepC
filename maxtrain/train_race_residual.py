import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

device = torch.device("cpu")

# ======================
# Model
# ======================

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


model = RaceCompensator().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()


race_dir = "dataset/race"
residual_dir = "dataset/residual"

epochs = 20

# ======================
# Training
# ======================

for epoch in range(epochs):

    total_loss = 0
    samples = 0

    for frame in range(150):
        for run in range(1,6):

            race = np.load(f"{race_dir}/frame_{frame:03d}_run{run}.npy")
            residual = np.load(f"{residual_dir}/frame_{frame:03d}_run{run}.npy")

            race = torch.tensor(race).float()/255.0
            residual = torch.tensor(residual).float()/255.0

            race = race.unsqueeze(0).unsqueeze(0).to(device)
            residual = residual.unsqueeze(0).unsqueeze(0).to(device)

            pred = model(race)

            loss = loss_fn(pred, residual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            samples += 1

    avg_loss = total_loss / samples

    print(f"Epoch {epoch+1} | Loss {avg_loss:.6f}")

# ======================
# Save model
# ======================

torch.save(model.state_dict(), "race_residual_compensator.pth")

print("Model saved")