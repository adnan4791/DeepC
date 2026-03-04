import numpy as np
import os
from glob import glob

def compute_noise_stats(data_root, num_samples=100):
    gt_files = sorted(glob(os.path.join(data_root, 'gt', '*.yuv')))
    errors = []
    for gt_path in gt_files[:num_samples]:  # batasi jumlah sampel
        basename = os.path.basename(gt_path)
        with open(gt_path, 'rb') as f:
            gt = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
        for run in range(30):
            run_path = os.path.join(data_root, f'run{run}', basename)
            if not os.path.exists(run_path):
                continue
            with open(run_path, 'rb') as f:
                run = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
            errors.extend(run - gt)
    errors = np.array(errors)
    mean = np.mean(errors)
    std = np.std(errors)
    print(f"Rata-rata noise: {mean:.6f}")
    print(f"Standar deviasi noise: {std:.6f}")
    return mean, std

if __name__ == '__main__':
    compute_noise_stats('train_data')