import numpy as np
import os

clean_dir = "dataset/clean"
race_dir = "dataset/race"
out_dir = "dataset/residual"

os.makedirs(out_dir, exist_ok=True)

frames = 150
runs = 5

for r in range(1, runs+1):

    run_out = f"{out_dir}"
    os.makedirs(run_out, exist_ok=True)

    for i in range(frames):

        clean = np.load(f"{clean_dir}/frame_{i:03d}.npy")
        race  = np.load(f"{race_dir}/frame_{i:03d}_run{r}.npy")

        residual = clean.astype(np.float32) - race.astype(np.float32)

        np.save(f"{run_out}/frame_{i:03d}_run{r}.npy", residual)

print("DONE generating residual dataset")

# VALIDASI
# import numpy as np

# r = np.load("dataset/residual/frame_000_run1.npy")

# print(r.min(), r.max(), r.mean())