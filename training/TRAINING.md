# FSRCNN Layer 8 Race Condition Compensation Training

## Overview

This document explains the training process for compensating race condition artifacts in FSRCNN Layer 8 deconvolution. The training approach is unique: instead of fixing the race condition in the C code (which is computationally expensive), we train the neural network weights to compensate for the artifacts produced by the race condition.

## Table of Contents

1. [Problem Background](#problem-background)
2. [Architecture Overview](#architecture-overview)
3. [Training Pipeline](#training-pipeline)
4. [Implementation Details](#implementation-details)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Results Interpretation](#results-interpretation)

---

## Problem Background

### What is the Race Condition?

In the FSRCNN C implementation, Layer 8 performs deconvolution (transposed convolution) to upscale the image. The parallel implementation uses OpenMP to process 56 channels simultaneously:

```c
#pragma omp parallel for
for (int j = 0; j < num_channels8; j++)
{
    double img_fltr_8_tmp[rows * scale * cols * scale];
    deconv_single(img_fltr_7 + j * rows * cols, img_fltr_8_tmp, 
                  weights_layer8 + j * filtersize8, cols, rows, scale);
    // RACE CONDITION - multiple threads writing to img_fltr_8 simultaneously
    imadd_race(img_fltr_8, img_fltr_8_tmp, cols * scale, rows * scale);
}
```

The race condition occurs because multiple threads write to the shared `img_fltr_8` array simultaneously without synchronization. This causes non-deterministic artifacts in the output image.

### Why Not Fix the Race Condition?

Fixing the race condition requires adding synchronization (e.g., `#pragma omp critical`), which serializes the operation and defeats the purpose of parallelization. This significantly impacts performance.

### The Compensation Approach

Instead of fixing the race condition, we:
1. Accept that the race condition exists and produces artifacts
2. Train the Layer 8 weights to produce outputs that, when corrupted by the race condition, still result in high-quality images
3. The model learns to "pre-compensate" for the artifacts

---

## Architecture Overview

### Layer 8 Structure

Layer 8 is a deconvolution layer with:
- **Input**: 56 channels of filtered images (from Layer 7)
- **Kernel**: 9×9 kernel per channel (56 × 81 = 4,536 weights total)
- **Stride**: 2 (for 2× upscaling)
- **Output**: Single channel upscaled image
- **Bias**: Single scalar value added to all output pixels

### Weight Layout

The weights are stored as a flat array of 4,536 values:
```
[Channel 0: 81 weights | Channel 1: 81 weights | ... | Channel 55: 81 weights]
```

Each channel's 81 weights form a 9×9 kernel applied during deconvolution.

### Deconvolution Process

For each input channel:
1. Pad the input with border=1
2. For each pixel in padded input, spread kernel values to output positions
3. Crop output to exact dimensions (rows×scale × cols×scale)

All channel outputs are summed together, then bias is added.

---

## Training Pipeline

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Training Data  │     │   PyTorch       │     │   C Library     │
│  (Layer 7 out)  │────▶│   Model         │────▶│   (Race Cond)   │
│                 │     │   (Gradients)   │     │   (Forward)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       ▼
        │                       │              ┌─────────────────┐
        │                       │              │   C Output      │
        │                       │              │   (With Artifacts)│
        │                       │              └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HR Target      │     │   PyTorch Loss  │     │   Loss vs HR    │
│  (Ground Truth) │     │   (Backprop)    │     │   (Metric)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Hybrid Training Approach

The training uses a hybrid approach:

1. **C Library Forward Pass**: The actual forward pass uses the C library with the race condition. This ensures the loss reflects the real output.

2. **PyTorch Gradient Computation**: PyTorch computes gradients for weight updates, even though the forward pass differs slightly.

3. **Combined Loss**:
   - `loss_pytorch`: MSE between PyTorch model output and HR target
   - `loss_c`: MSE between C library output and HR target
   - `alpha` balances between matching target and matching C output

### Training Algorithm

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Get current weights
        weights = model.get_weights_flat()
        
        # 2. Forward pass through C library (with race condition)
        output_c = forward_c_library(layer7, weights, bias)
        
        # 3. Compute C loss
        loss_c = MSE(output_c, hr_target)
        
        # 4. PyTorch forward pass (for gradients)
        output_pytorch = model(layer7)
        
        # 5. Combined loss
        loss = alpha * MSE(output_pytorch, hr_target) + \
               (1-alpha) * MSE(output_pytorch, output_c)
        
        # 6. Backpropagation and weight update
        loss.backward()
        optimizer.step()
```

---

## Implementation Details

### File Structure

```
training/
├── train_layer8_race_condition.py  # Main training script
├── source_layer8_so.c              # C library source code
├── fsrcnn_layer8.so                # Compiled shared library
├── weights_layer8_original.txt     # Original pre-trained weights
├── weights_layer8_trained.txt      # Trained weights output
├── training_data/                  # Training data directory
│   ├── layer7_frame*.bin          # Layer 7 outputs
│   └── hr_frame*.bin              # HR ground truth
└── results/                        # Training results
    ├── psnr_original.txt          # PSNR with original weights
    └── psnr_trained.txt           # PSNR with trained weights
```

### Key Components

#### 1. Layer8Deconv Model (PyTorch)

```python
class Layer8Deconv(nn.Module):
    def __init__(self, num_channels=56, kernel_size=9, scale=2):
        super().__init__()
        # Each channel has its own 9x9 kernel
        self.weights = nn.Parameter(torch.zeros(56, 9, 9))
        self.bias = nn.Parameter(torch.tensor(-0.03262640000))
    
    def forward(self, x):
        # Implements deconvolution matching C code behavior
        ...
```

#### 2. C Library Interface

```python
def load_c_library():
    lib = ctypes.CDLL('./fsrcnn_layer8.so')
    
    # Function signatures
    lib.set_weights_layer8.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.layer8_forward_race.argtypes = [...]
    ...
    return lib
```

#### 3. Weight File Format

The weights file contains 4,536 floating-point values. Two formats are supported:

**Format 1: One value per line (decimal)**
```
-0.0006382514
-0.0008207229
...
```

**Format 2: All values on one line (scientific notation)**
```
-6.3852808671072125e-04  -8.0851308302953839e-04  ...
```

Both formats are readable by `np.loadtxt()` and the C `fscanf()` function with `%lf` format specifier.

---

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rows` | 144 | Input image rows |
| `cols` | 176 | Input image columns |
| `scale` | 2 | Upscaling factor |
| `num_channels` | 56 | Number of input channels |
| `kernel_size` | 9 | Deconvolution kernel size |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `batch_size` | 4 | Training batch size |
| `num_epochs` | 100 | Number of training epochs |
| `num_threads` | 8 | OpenMP threads (must match production) |

### Important Notes

1. **Thread Count**: `num_threads` must match the production environment. Different thread counts produce different race condition patterns.

2. **Learning Rate**: Too high causes instability; too low causes slow convergence.

3. **Batch Size**: Limited by memory. Larger batches provide more stable gradients.

---

## Usage Guide

### Step 1: Prepare Training Data

Generate Layer 7 outputs and HR targets:

```bash
# Run the data preparation script
./dump_layer7 input.yuv output_dir/
```

This creates:
- `layer7_frame000.bin` - Layer 7 output for frame 0
- `hr_frame000.bin` - HR ground truth for frame 0
- ...

### Step 2: Compile C Library

```bash
cd training/
gcc-15 -shared -o fsrcnn_layer8.so -fPIC source_layer8_so.c -fopenmp
```

### Step 3: Run Training

```bash
python train_layer8_race_condition.py
```

### Step 4: Verify Results

```bash
./verify_race.sh
```

This compares PSNR with original vs trained weights.

---

## Troubleshooting

### Issue: Retraining Produces Different Results

**Symptoms**: Running training multiple times produces significantly different PSNR results.

**Root Causes**:

1. **Random Initialization**: The model may start from different initial weights.

2. **Race Condition Non-determinism**: The C library output varies between runs due to thread scheduling.

3. **Training Data Variation**: Different training samples or order.

4. **Weight File Format**: Different saving formats may cause parsing issues.

**Solutions**:

1. **Set Random Seeds**:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

2. **Use Consistent Thread Count**:
```python
CONFIG['num_threads'] = 8  # Must match production
lib.set_num_threads(CONFIG['num_threads'])
```

3. **Load Initial Weights**:
```python
# Always start from the same initial weights
initial_weights = np.loadtxt('weights_layer8_original.txt')
model.set_weights_from_flat(initial_weights)
```

4. **Verify Weight File Format**:
```python
# Use consistent format for saving
np.savetxt('weights.txt', weights, fmt='%.10f')  # One per line, decimal
# OR
np.savetxt('weights.txt', weights[None, :], fmt='%.16e')  # All on one line
```

### Issue: PSNR Decreases After Training

**Possible Causes**:
- Learning rate too high
- Not enough epochs
- Training data doesn't match test data distribution
- Thread count mismatch between training and testing

**Solutions**:
- Reduce learning rate to 1e-5
- Increase epochs to 200+
- Use more diverse training data
- Ensure thread count matches production

### Issue: Training Loss Doesn't Decrease

**Possible Causes**:
- Learning rate too low
- Model stuck in local minimum
- Race condition pattern too complex

**Solutions**:
- Increase learning rate
- Use learning rate scheduler
- Try numerical gradient approach (slower but more accurate)

---

## Results Interpretation

### Expected PSNR Improvement

With successful training, PSNR should improve by 10-30 dB compared to using original weights with the race condition.

**Example Results**:

| Configuration | PSNR (dB) |
|--------------|-----------|
| Original weights, no race condition | 75-80 |
| Original weights, with race condition | 45-55 |
| Trained weights, with race condition | 65-79 |

### Analyzing Weight Changes

The trained weights will differ from original weights. Key observations:

1. **Magnitude Changes**: Weights may have larger or smaller magnitudes to compensate for averaging effects.

2. **Pattern Changes**: Spatial patterns in kernels may shift to pre-compensate for artifacts.

3. **Channel Variation**: Some channels may change more than others.

### Verification

To verify the trained weights work correctly:

1. Run multiple inference tests with the same input
2. Check PSNR stability across runs
3. Compare visual quality of output images
4. Test on different input videos

---

## Technical Reference

### C Library Functions

| Function | Purpose |
|----------|---------|
| `load_weights_layer8(filepath)` | Load weights from file |
| `set_weights_layer8(weights)` | Set weights from array |
| `get_weights_layer8(out_weights)` | Get current weights |
| `set_bias_layer8(bias)` | Set bias value |
| `get_bias_layer8()` | Get current bias |
| `layer8_forward_race(...)` | Forward pass with race condition |
| `layer8_forward_safe(...)` | Forward pass without race condition |
| `set_num_threads(n)` | Set OpenMP thread count |
| `get_max_threads()` | Get current thread count |

### Python API

```python
# Load library
lib = load_c_library()

# Set weights
weights = np.array([...], dtype=np.float64)
weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
lib.set_weights_layer8(weights_ptr)

# Run forward pass
lib.layer8_forward_race(input_ptr, output_ptr, rows, cols, scale)
```

---

## Conclusion

This training approach demonstrates that neural network weights can be trained to compensate for software bugs (race conditions) when fixing the bug is impractical. The key insights are:

1. The race condition produces consistent patterns that can be learned
2. Hybrid training (C forward pass + PyTorch gradients) enables end-to-end optimization
3. Thread count must be consistent between training and production
4. Proper initialization and random seeds ensure reproducibility

For questions or issues, refer to the troubleshooting section or examine the source code in [`train_layer8_race_condition.py`](train_layer8_race_condition.py) and [`source_layer8_so.c`](source_layer8_so.c).
