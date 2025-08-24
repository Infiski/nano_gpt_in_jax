# GPT-2 (124M) in JAX/Flax

## Overview
This project reproduces the **GPT-2 (124M)** model using **JAX/Flax**, with a focus on leveraging advanced training optimizations to achieve high throughput on modern GPUs.  
Compared to the PyTorch baseline, the JAX implementation achieves up to **40× faster training throughput** on the same NVIDIA L4 GPU.

## Key Features
- **Reimplementation of GPT-2 (124M)** in JAX/Flax with functional parity to PyTorch.  
- **Advanced optimizations**:
  - JIT compilation for compiled execution.  
  - Gradient accumulation for memory efficiency.  
  - Multi-GPU parallelization for scaling training across devices.  
- **Performance Gains**: Up to ~40× throughput improvement over PyTorch baselines.  

## Results
| Framework | GPU  | Tokens/sec (approx.) | Speedup |
|-----------|------|-----------------------|---------|
| PyTorch   | L4   | ~1k tokens/s          | 1×      |
| JAX/Flax  | L4   | ~40k tokens/s         | ~40×    |

## Requirements
- Python 3.10+  
- [JAX](https://github.com/google/jax) with GPU support (`pip install jax jaxlib`)  
- [Flax](https://github.com/google/flax)  
- [Optax](https://github.com/deepmind/optax)  
- CUDA/cuDNN (if using NVIDIA GPUs)  

