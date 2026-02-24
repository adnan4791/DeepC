import matplotlib.pyplot as plt
import numpy as np

import os

# Data Loading Logic
def load_psnr(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using empty data.")
        return []
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

psnr_trained = load_psnr('psnr_trained.txt')
psnr_original = load_psnr('psnr_original.txt')

# Baseline (Before Training) average
if psnr_original:
    baseline_avg = sum(psnr_original) / len(psnr_original)
else:
    baseline_avg = 49.49  # Fallback

plt.figure(figsize=(12, 6))

# Get actual number of iterations
num_iterations = len(psnr_trained)
x_range = range(1, num_iterations + 1)

# Plot Compensated data
plt.plot(x_range, psnr_trained, color='#2ca02c', marker='o', markersize=4, linestyle='-', linewidth=1, label='After Training (Compensated)')

# Plot Baseline average line
plt.axhline(y=baseline_avg, color='#d62728', linestyle='--', linewidth=2, label=f'Before Training Avg (~{baseline_avg:.2f} dB)')

# Fill area below baseline to highlight improvement
plt.fill_between(x_range, baseline_avg, psnr_trained, where=(np.array(psnr_trained) > baseline_avg), color='#2ca02c', alpha=0.1)

# Annotate the worst case
if psnr_trained:
    min_idx = np.argmin(psnr_trained)
    min_val = psnr_trained[min_idx]
    plt.annotate(f'Worst Case (Run {min_idx + 1})\n{min_val:.2f} dB', 
                 xy=(min_idx + 1, min_val), xytext=(min_idx + 1, min_val - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.title('FSRCNN PSNR Stability: Before vs After Training (100 Runs)', fontsize=14)
plt.xlabel('Run Iteration', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

# Set y-axis to show the clear gap
plt.ylim(40, 105)

plt.tight_layout()
plt.savefig('psnr_stability_analysis.png')
print("Graph saved as psnr_stability_analysis.png")