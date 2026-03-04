import numpy as np
import os
import random

def load_noise_pool(gt_dir, runs_dir, num_frames=150, hr_shape=(288,352)):
    """Mengembalikan list noise per piksel dari semua run."""
    noise_pool = []
    for f in range(num_frames):
        gt_path = os.path.join(gt_dir, f'frame_{f:04d}.yuv')
        gt = np.fromfile(gt_path, dtype=np.uint8).reshape(hr_shape).astype(np.float32) / 255.0
        for run in range(30):
            run_path = os.path.join(runs_dir, f'run{run}', f'frame_{f:04d}.yuv')
            run = np.fromfile(run_path, dtype=np.uint8).reshape(hr_shape).astype(np.float32) / 255.0
            noise = run - gt
            noise_pool.append(noise.flatten())
    noise_pool = np.concatenate(noise_pool)
    return noise_pool

def generate_noise_batch(noise_pool, batch_size, img_size):
    """Mengembalikan batch noise acak dari noise_pool."""
    n_pixels = img_size[0] * img_size[1]
    batch = np.zeros((batch_size, n_pixels))
    for i in range(batch_size):
        idx = random.randint(0, len(noise_pool) - 1)
        batch[i] = noise_pool[idx]
    return batch

if __name__ == '__main__':
    noise_pool = load_noise_pool('train_data/gt', 'train_data', num_frames=150, hr_shape=(288,352))
    print(f"Noise pool shape: {noise_pool.shape}")
    batch = generate_noise_batch(noise_pool, 16, (288,352))
    print(f"Batch shape: {batch.shape}")