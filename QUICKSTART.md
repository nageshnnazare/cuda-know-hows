# CUDA Tutorial - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This is a comprehensive CUDA programming tutorial covering everything from basics to advanced topics.

### What You'll Learn

```
Basic Level          Intermediate Level       Advanced Level
─────────────        ──────────────────       ──────────────
✓ GPU Architecture   ✓ Matrix Operations      ✓ Streams
✓ First Kernel       ✓ Shared Memory          ✓ Atomics  
✓ Memory Model       ✓ Optimization           ✓ Dynamic Parallelism
✓ Thread Indexing                             ✓ Profiling Tools
```

---

## Prerequisites

✅ **Required:**
- NVIDIA GPU (Compute Capability ≥ 3.0)
- CUDA Toolkit installed
- Basic C/C++ knowledge

✅ **Check Your Setup:**
```bash
# Verify CUDA installation
nvcc --version

# Check GPU
nvidia-smi
```

---

## Quick Test

```bash
# Navigate to the directory
cd /tmp/cuda

# Build an example
make 02_first_kernel

# Run it
./02_first_kernel
```

**Expected Output:**
```
╔═══════════════════════════════════════════════════════╗
║       CUDA Tutorial: First Kernel Examples           ║
╚═══════════════════════════════════════════════════════╝

📊 Device Information:
   Name: NVIDIA GeForce RTX 3080
   ...

✓ Results verified successfully!
GPU time: 2.145 ms
Speedup: 21.27x
```

---

## Tutorial Structure

### 📁 Files Overview

| File | Description | Level |
|------|-------------|-------|
| `01_introduction.md` | GPU concepts & architecture | Beginner |
| `02_first_kernel.cu` | Your first CUDA program | Beginner |
| `03_memory_model.cu` | Memory hierarchy & access | Beginner |
| `04_thread_organization.cu` | Thread indexing (1D/2D/3D) | Beginner |
| `05_matrix_operations.cu` | Matrix multiplication | Intermediate |
| `06_shared_memory.cu` | Advanced memory optimization | Intermediate |
| `07_streams_async.cu` | Concurrent execution | Advanced |
| `08_advanced_topics.cu` | Atomics, warps, dynamic parallelism | Advanced |
| `09_profiling_debugging.md` | Profiling & debugging tools | All Levels |
| `README.md` | Complete documentation | Reference |
| `Makefile` | Build system | Tool |

---

## Learning Path

### Week 1: Foundations
1. Read `01_introduction.md` (30 min)
2. Build and run `02_first_kernel.cu` (1 hour)
3. Experiment with `03_memory_model.cu` (1.5 hours)
4. Practice `04_thread_organization.cu` (1 hour)

**Goal:** Understand GPU basics and write simple kernels.

### Week 2: Practical Applications
1. Study `05_matrix_operations.cu` (2 hours)
2. Master `06_shared_memory.cu` (2 hours)
3. Build your own examples

**Goal:** Implement and optimize real algorithms.

### Week 3: Advanced Techniques
1. Learn `07_streams_async.cu` (1.5 hours)
2. Explore `08_advanced_topics.cu` (2.5 hours)
3. Read `09_profiling_debugging.md` (2 hours)

**Goal:** Use advanced features and optimization tools.

---

## Common Commands

### Building
```bash
# Build all examples
make all

# Build specific example
make 05_matrix_operations

# Clean
make clean

# Build with debug symbols
make debug
```

### Running
```bash
# Run single example
./02_first_kernel

# Run all examples
make run
```

### Profiling
```bash
# Profile with Nsight Systems (timeline)
make profile-05_matrix_operations

# Analyze with Nsight Compute (kernel details)
make analyze-05_matrix_operations

# Check for memory errors
make memcheck-06_shared_memory
```

### Debugging
```bash
# Build with debug info
make debug

# Debug with cuda-gdb
cuda-gdb ./02_first_kernel

# Check memory errors
cuda-memcheck ./02_first_kernel
```

---

## Help & Troubleshooting

### Installation Issues

**Problem:** `nvcc: command not found`
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem:** `no CUDA-capable device is detected`
```bash
# Check GPU
lspci | grep -i nvidia

# Check driver
nvidia-smi

# May need to install/update driver
```

### Build Issues

**Problem:** `cannot find -lcublas`
```bash
# Install CUDA development libraries
sudo apt-get install nvidia-cuda-toolkit
```

**Problem:** Compilation errors
```bash
# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update ARCH in Makefile
# Edit Makefile: ARCH := sm_75  (for your GPU)
```

### Runtime Issues

**Problem:** Slow performance
```bash
# Profile to find bottlenecks
make profile-<example>

# Check occupancy and memory efficiency
make analyze-<example>
```

**Problem:** Wrong results
```bash
# Check for memory errors
cuda-memcheck ./program

# Debug with cuda-gdb
cuda-gdb ./program
```

---

## Quick Reference

### Memory Management
```c
float *d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// ... use data ...
cudaFree(d_data);
```

### Kernel Launch
```c
// 1D
kernel<<<(n+255)/256, 256>>>(args);

// 2D
dim3 block(16, 16);
dim3 grid((w+15)/16, (h+15)/16);
kernel<<<grid, block>>>(args);
```

### Error Checking
```c
// Always check errors!
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

### Thread Indexing
```c
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## Next Steps

1. **Start Learning:**
   - Begin with `01_introduction.md`
   - Follow the tutorial in order

2. **Practice:**
   - Modify the examples
   - Write your own kernels
   - Experiment with parameters

3. **Optimize:**
   - Profile your code
   - Apply optimization techniques
   - Compare performance

4. **Explore:**
   - Read NVIDIA documentation
   - Join CUDA community
   - Build real projects

---

## Resources

### Documentation
- [Full README](README.md) - Complete guide
- [Tutorial Part 1](01_introduction.md) - Start here
- [Profiling Guide](09_profiling_debugging.md) - Debug & optimize

### Official Links
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CUDA C++ Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Community
- [NVIDIA Forums](https://forums.developer.nvidia.com/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/cuda)
- [r/CUDA](https://www.reddit.com/r/CUDA/)

---

## Makefile Targets

```bash
make help              # Show all available commands
make info              # Display GPU information
make install-check     # Verify tool installation
make test              # Quick sanity test
make list              # List all examples
```

---

## Tips for Success

✅ **DO:**
- Start with the basics
- Check all return values
- Run cuda-memcheck regularly
- Profile before optimizing
- Read error messages carefully

❌ **DON'T:**
- Skip error checking
- Optimize prematurely
- Forget boundary checks
- Ignore profiler results
- Give up on first error

---

## Summary

This tutorial provides:
- ✅ 8 complete code examples
- ✅ 2 comprehensive guides
- ✅ Graphics and visualizations
- ✅ Build system (Makefile)
- ✅ Profiling instructions
- ✅ Debugging techniques

**Total Time:** 20-30 hours for complete mastery

**Start here:** [01_introduction.md](01_introduction.md)

---

## Getting Help

1. Check the [README.md](README.md)
2. Review [09_profiling_debugging.md](09_profiling_debugging.md)
3. Search NVIDIA forums
4. Ask on Stack Overflow with tag [cuda]

---

**Happy CUDA Programming! 🚀**

*Start your journey: `make 02_first_kernel && ./02_first_kernel`*

