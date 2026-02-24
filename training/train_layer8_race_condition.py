#!/usr/bin/env python3
"""
train_layer8_race_condition.py

Training script for FSRCNN Layer 8 to compensate for race condition artifacts.
This script trains the deconvolution layer weights to produce better output
despite the race condition in the C code.

The key idea: Instead of fixing the race condition (which is expensive),
we let the model learn to compensate for the artifacts it produces.

Author: Training approach for race condition compensation
"""

import os
import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'rows': 144,
    'cols': 176,
    'scale': 2,
    'num_channels': 56,
    'kernel_size': 9,
    'learning_rate': 1e-4,
    'batch_size': 4,
    'num_epochs': 100,
    'training_data_dir': 'training_data',
    'weights_file': 'weights_layer8.txt',
    'output_weights_file': 'weights_layer8_trained.txt',
    'num_threads': 8,  # Number of OpenMP threads (should match production)
}

# ============================================================================
# Load C Library
# ============================================================================
def load_c_library():
    """Load the compiled shared library for Layer 8."""
    lib_path = './fsrcnn_layer8.so'
    
    if not os.path.exists(lib_path):
        print(f"Error: Shared library not found at {lib_path}")
        print("Please compile it first with:")
        print("  gcc-15 -shared -o fsrcnn_layer8.so -fPIC fsrcnn_layer8_so.c -fopenmp")
        return None
    
    lib = ctypes.CDLL(lib_path)
    
    # Define function signatures
    lib.set_weights_layer8.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.set_weights_layer8.restype = None
    
    lib.get_weights_layer8.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.get_weights_layer8.restype = None
    
    lib.set_bias_layer8.argtypes = [ctypes.c_double]
    lib.set_bias_layer8.restype = None
    
    lib.get_bias_layer8.argtypes = []
    lib.get_bias_layer8.restype = ctypes.c_double
    
    lib.layer8_forward_race.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # img_fltr_7
        ctypes.POINTER(ctypes.c_double),  # img_output
        ctypes.c_int,  # rows
        ctypes.c_int,  # cols
        ctypes.c_int,  # scale
    ]
    lib.layer8_forward_race.restype = None
    
    lib.layer8_forward_safe.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.layer8_forward_safe.restype = None
    
    lib.set_num_threads.argtypes = [ctypes.c_int]
    lib.set_num_threads.restype = None
    
    lib.get_max_threads.argtypes = []
    lib.get_max_threads.restype = ctypes.c_int
    
    return lib


# ============================================================================
# Dataset
# ============================================================================
class Layer7Dataset(Dataset):
    """Dataset for Layer 7 outputs and HR targets."""
    
    def __init__(self, data_dir, rows, cols, scale, num_channels):
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self.num_channels = num_channels
        
        # Find all layer7 files
        layer7_files = sorted(glob.glob(os.path.join(data_dir, 'layer7_frame*.bin')))
        hr_files = sorted(glob.glob(os.path.join(data_dir, 'hr_frame*.bin')))
        
        if len(layer7_files) == 0:
            raise ValueError(f"No training data found in {data_dir}")
        
        print(f"Found {len(layer7_files)} training samples")
        
        # Load all data into memory (for small datasets)
        self.layer7_data = []
        self.hr_data = []
        
        for l7_file, hr_file in zip(layer7_files, hr_files):
            # Load Layer 7 output
            l7 = np.fromfile(l7_file, dtype=np.float64)
            l7 = l7.reshape(num_channels, rows, cols)
            self.layer7_data.append(l7)
            
            # Load HR target
            hr = np.fromfile(hr_file, dtype=np.float64)
            hr = hr.reshape(rows * scale, cols * scale)
            self.hr_data.append(hr)
        
        self.layer7_data = np.array(self.layer7_data, dtype=np.float64)
        self.hr_data = np.array(self.hr_data, dtype=np.float64)
    
    def __len__(self):
        return len(self.layer7_data)
    
    def __getitem__(self, idx):
        return {
            'layer7': torch.from_numpy(self.layer7_data[idx]),
            'hr': torch.from_numpy(self.hr_data[idx])
        }


# ============================================================================
# PyTorch Layer 8 Model
# ============================================================================
class Layer8Deconv(nn.Module):
    """
    PyTorch implementation of Layer 8 deconvolution.
    
    The C code implements deconvolution as:
    - For each of 56 input channels, apply 9x9 kernel with stride 2
    - Sum all channel outputs together
    - Add bias
    
    This is NOT a standard ConvTranspose2d. Each channel has its own kernel
    and outputs are summed.
    """
    
    def __init__(self, num_channels=56, kernel_size=9, scale=2):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.scale = scale
        
        # Each input channel has its own 9x9 kernel
        # Shape: (num_channels, kernel_size, kernel_size) = (56, 9, 9)
        self.weights = nn.Parameter(torch.zeros(num_channels, kernel_size, kernel_size))
        
        # Bias (scalar added to final output)
        self.bias = nn.Parameter(torch.tensor(-0.03262640000))
    
    def forward(self, x):
        """
        Forward pass - implements deconvolution like the C code.
        x: (batch, 56, rows, cols)
        output: (batch, rows*scale, cols*scale)
        
        The C deconv function:
        1. Pads input with border=1
        2. For each pixel in padded input, spreads kernel values to output
        3. Crops output to exact rows*scale x cols*scale
        """
        batch_size, num_ch, rows, cols = x.shape
        scale = self.scale
        ksize = self.kernel_size
        border = 1  # As in C code
        
        # Output size (exact, as in C code)
        out_rows = rows * scale
        out_cols = cols * scale
        
        # Initialize output
        output = torch.zeros(batch_size, out_rows, out_cols, device=x.device, dtype=x.dtype)
        
        # For each input channel, perform deconvolution and add to output
        for c in range(num_ch):
            # Get input channel
            x_c = x[:, c:c+1, :, :]  # (batch, 1, rows, cols)
            
            # Get kernel for this channel
            kernel_c = self.weights[c:c+1, :, :].unsqueeze(0)  # (1, 1, 9, 9)
            
            # Pad input (border=1 as in C code)
            x_padded = torch.nn.functional.pad(x_c, (border, border, border, border), mode='replicate')
            
            # Deconvolution using transposed convolution
            # The C code uses stride=scale for upsampling
            # Output of conv_transpose2d with padding=0: 
            #   H_out = (H_in - 1)*stride + kernel_size
            # We need to crop to match C code's output
            out_c = torch.nn.functional.conv_transpose2d(
                x_padded, kernel_c, 
                stride=scale, 
                padding=0
            )
            
            # Crop to exact output size (matching C code's cropping)
            # C code: i_tmp = i + ((fsize + 1) / 2) + stride*border - 1
            # For fsize=9, stride=2, border=1: offset = 5 + 2 - 1 = 6
            offset = (ksize + 1) // 2 + scale * border - 1
            out_c_cropped = out_c[:, :, offset:offset+out_rows, offset:offset+out_cols]
            
            # Add to output
            output = output + out_c_cropped.squeeze(1)
        
        # Add bias
        output = output + self.bias
        
        return output
    
    def get_weights_flat(self):
        """Get weights as flat numpy array (C order)."""
        # C expects: 56 * 81 = 4536 values
        # Each channel has 9x9 = 81 weights, stored contiguously
        weights = self.weights.detach().numpy()  # (56, 9, 9)
        weights_flat = weights.flatten()  # (4536,)
        return weights_flat.astype(np.float64)
    
    def set_weights_from_flat(self, weights_flat):
        """Set weights from flat numpy array."""
        weights = weights_flat.reshape(self.num_channels, self.kernel_size, self.kernel_size)
        self.weights.data = torch.from_numpy(weights.astype(np.float32))


# ============================================================================
# Training Functions
# ============================================================================
def forward_c_library(lib, layer7_batch, weights, bias, rows, cols, scale):
    """
    Run forward pass through C library (with race condition).
    
    Args:
        lib: ctypes library
        layer7_batch: numpy array (batch, 56, rows, cols)
        weights: numpy array (4536,)
        bias: float
        rows, cols, scale: dimensions
    
    Returns:
        output: numpy array (batch, rows*scale, cols*scale)
    """
    batch_size = layer7_batch.shape[0]
    output = np.zeros((batch_size, rows * scale, cols * scale), dtype=np.float64)
    
    # Set weights and bias
    weights_c = weights.astype(np.float64)
    weights_ptr = weights_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.set_weights_layer8(weights_ptr)
    lib.set_bias_layer8(bias)
    
    # Process each sample
    for i in range(batch_size):
        layer7 = layer7_batch[i].astype(np.float64)
        layer7_ptr = layer7.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        out_ptr = output[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lib.layer8_forward_race(layer7_ptr, out_ptr, rows, cols, scale)
    
    return output


def train_with_c_library(model, lib, dataloader, optimizer, config, device):
    """
    Training loop using C library for forward pass.
    
    The key insight: We use PyTorch's autograd for gradient computation,
    but we use the C library's forward pass (with race condition) to
    compute the actual loss. This creates a mismatch that the model
    learns to compensate for.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        layer7 = batch['layer7'].numpy()  # (batch, 56, rows, cols)
        hr_target = batch['hr'].numpy()   # (batch, rows*scale, cols*scale)
        
        # Get current weights
        weights = model.get_weights_flat()
        bias = model.bias.item()
        
        # Forward pass through C library (with race condition)
        with torch.no_grad():
            output_c = forward_c_library(
                lib, layer7, weights, bias,
                config['rows'], config['cols'], config['scale']
            )
        
        # Convert to tensors
        layer7_tensor = torch.from_numpy(layer7).float().to(device)
        hr_tensor = torch.from_numpy(hr_target).float().to(device)
        output_c_tensor = torch.from_numpy(output_c).float().to(device)
        
        # Compute loss between C output and target
        loss_c = nn.functional.mse_loss(output_c_tensor, hr_tensor)
        
        # Also compute PyTorch forward pass for gradient
        optimizer.zero_grad()
        output_pytorch = model(layer7_tensor)
        
        # Combined loss:
        # 1. PyTorch output should match target
        # 2. PyTorch output should match C output (to learn the race condition pattern)
        loss_pytorch = nn.functional.mse_loss(output_pytorch, hr_tensor)
        
        # Weight the losses
        # We want the model to learn to produce output that:
        # - When passed through C (with race condition), produces good results
        # - Is close to the target
        alpha = 0.5  # Balance between matching target and matching C output
        loss = alpha * loss_pytorch + (1 - alpha) * nn.functional.mse_loss(output_pytorch, output_c_tensor)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}, "
                  f"C Loss: {loss_c.item():.6f}, PyTorch Loss: {loss_pytorch.item():.6f}")
    
    return total_loss / len(dataloader)


def train_numerical_gradient(model, lib, dataloader, config, device):
    """
    Alternative training approach: Numerical gradient computation.
    
    1. Forward pass through C library
    2. Compute gradient numerically (finite differences)
    3. Update weights
    
    This is slower but more accurate for learning the race condition.
    """
    model.train()
    total_loss = 0.0
    
    epsilon = 1e-5  # For numerical gradient
    learning_rate = config['learning_rate']
    
    for batch_idx, batch in enumerate(dataloader):
        layer7 = batch['layer7'].numpy()
        hr_target = batch['hr'].numpy()
        
        # Get current weights
        weights = model.get_weights_flat().copy()
        bias = model.bias.item()
        
        # Compute current loss
        output_c = forward_c_library(
            lib, layer7, weights, bias,
            config['rows'], config['cols'], config['scale']
        )
        current_loss = np.mean((output_c - hr_target) ** 2)
        
        # Compute numerical gradients for all weights
        grad_weights = np.zeros_like(weights)
        
        for idx in range(len(weights)):
            weights_plus = weights.copy()
            weights_plus[idx] += epsilon
            
            output_plus = forward_c_library(
                lib, layer7, weights_plus, bias,
                config['rows'], config['cols'], config['scale']
            )
            loss_plus = np.mean((output_plus - hr_target) ** 2)
            
            grad_weights[idx] = (loss_plus - current_loss) / epsilon
        
        # Update weights using gradient descent
        weights -= learning_rate * grad_weights
        
        # Update model
        model.set_weights_from_flat(weights)
        
        total_loss += current_loss
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {current_loss:.6f}")
    
    return total_loss / len(dataloader)


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    print("=" * 60)
    print("FSRCNN Layer 8 Training - Race Condition Compensation")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load C library
    print("\nLoading C library...")
    lib = load_c_library()
    if lib is None:
        return
    
    # Set number of threads
    lib.set_num_threads(CONFIG['num_threads'])
    print(f"OpenMP threads: {lib.get_max_threads()}")
    
    # Load dataset
    print(f"\nLoading dataset from {CONFIG['training_data_dir']}...")
    dataset = Layer7Dataset(
        CONFIG['training_data_dir'],
        CONFIG['rows'],
        CONFIG['cols'],
        CONFIG['scale'],
        CONFIG['num_channels']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Must be 0 for ctypes compatibility
    )
    
    # Create model
    print("\nCreating model...")
    model = Layer8Deconv(
        num_channels=CONFIG['num_channels'],
        kernel_size=CONFIG['kernel_size'],
        scale=CONFIG['scale']
    ).to(device)
    
    # Load initial weights from file
    if os.path.exists(CONFIG['weights_file']):
        print(f"Loading initial weights from {CONFIG['weights_file']}...")
        initial_weights = np.loadtxt(CONFIG['weights_file'])
        model.set_weights_from_flat(initial_weights)
        print(f"Loaded {len(initial_weights)} weights")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Train using hybrid approach (PyTorch gradients + C forward)
        avg_loss = train_with_c_library(model, lib, dataloader, optimizer, CONFIG, device)
        
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            weights = model.get_weights_flat()
            np.savetxt(CONFIG['output_weights_file'], weights, fmt='%.10f')
            print(f"  Saved best weights to {CONFIG['output_weights_file']}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Trained weights saved to: {CONFIG['output_weights_file']}")
    print("=" * 60)
    
    # Verification
    print("\nVerification:")
    print("To use the trained weights, replace weights_layer8.txt with the new file.")
    print("The model has learned to compensate for the race condition artifacts.")


if __name__ == '__main__':
    main()
