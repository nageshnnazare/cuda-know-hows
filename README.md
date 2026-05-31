# Complete CUDA Programming Tutorial
## From Basics to Advanced Techniques

Welcome to the comprehensive CUDA programming tutorial! This tutorial series takes you from basic GPU concepts to advanced optimization techniques, with detailed explanations, graphics, and practical examples.

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗██╗   ██╗██████╗  █████╗     ████████╗██╗   ██╗     ║
║  ██╔════╝██║   ██║██╔══██╗██╔══██╗    ╚══██╔══╝██║   ██║     ║
║  ██║     ██║   ██║██║  ██║███████║       ██║   ██║   ██║     ║
║  ██║     ██║   ██║██║  ██║██╔══██║       ██║   ██║   ██║     ║
║  ╚██████╗╚██████╔╝██████╔╝██║  ██║       ██║   ╚██████╔╝     ║
║   ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝       ╚═╝    ╚═════╝      ║
║                                                              ║
║              Complete Tutorial Series                        ║
║           Basic → Intermediate → Advanced                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📚 Table of Contents

### Part 1: Foundations
1. [Introduction to CUDA](#part-1-introduction)
2. [Your First CUDA Kernel](#part-2-first-kernel)
3. [CUDA Memory Model](#part-3-memory-model)
4. [Thread Organization](#part-4-thread-organization)

### Part 2: Practical Applications
5. [Matrix Operations](#part-5-matrix-operations)
6. [Shared Memory Optimization](#part-6-shared-memory)
7. [Thread Indexing Patterns (1D/2D/3D)](#part-7-thread-indexing)

### Part 3: Advanced Topics
8. [Streams and Async Operations](#part-8-streams)
9. [Advanced Programming Techniques](#part-9-advanced-topics)
10. [Profiling and Debugging](#part-10-profiling-debugging)
11. [GPU Architecture Internals](#part-11-architecture-internals)
12. [Work Allocation & Execution Guide](#part-12-work-allocation) - Block/Grid sizing, occupancy

### Part 4: Practical Examples & Applications
13. [Image Processing](#part-13-image-processing) - Convolution, edge detection, filters
14. [Sorting Algorithms](#part-14-sorting) - Bitonic, radix, merge sort
15. [Scientific Computing](#part-15-scientific) - PDEs, N-body, Monte Carlo
16. [Optimization Case Studies](#part-16-optimization) - Before/after improvements
17. [Graph Algorithms](#part-17-graphs)
18. [Advanced Memory Techniques](#part-18-advanced-memory)
19. [Machine Learning Primitives](#part-19-ml-primitives)
20. [Multi-GPU Programming](#part-20-multi-gpu)
21. [Testing & Debugging Guide](#part-21-testing)
22. [Deep Learning from Scratch](#part-22-deep-learning) - Regression to CNNs

### Part 5: Modern CUDA (2024-2026)
23. [Modern CUDA & Blackwell](#part-23-modern-cuda) - CUDA Tile, FP4/FP8, Blackwell architecture

### Special Guides
- [Memory Allocation Guide](MEMORY_ALLOCATION_GUIDE.md) - Complete reference for host/device memory strategies
- [Work Allocation Guide](WORK_ALLOCATION_GUIDE.md) - Block/grid sizing, occupancy optimization
- [GPU Locks & Synchronization Guide](GPU_LOCKS_AND_SYNCHRONIZATION_GUIDE.md) - Atomics, sync primitives

### Additional Resources
- [Setup Instructions](#setup-instructions)
- [Building the Examples](#building-examples)
- [Quick Reference](#quick-reference)
- [Troubleshooting](#troubleshooting)
- [Further Reading](#further-reading)

---

## 🚀 Quick Start

```bash
# Clone or navigate to the tutorial directory
cd /tmp/cuda

# Build all examples
make all

# Run a specific example
./02_first_kernel

# Or run with profiling
nsys profile -o report ./02_first_kernel
```

---

## 📖 Tutorial Structure

### Part 1: Introduction
**File:** `01_introduction.md`

**Topics Covered:**
- What is CUDA and why use it?
- GPU vs CPU architecture
- CUDA programming model
- Memory hierarchy overview
- Setting up your environment

**Prerequisites:** None

**Visual Highlights:**
```
CPU Architecture          GPU Architecture
┌─────────────┐          ┌──────────────────┐
│ Few cores   │          │ Thousands cores  │
│ Complex     │    vs    │ Simple           │
│ Sequential  │          │ Parallel         │
└─────────────┘          └──────────────────┘
```

**Time to Complete:** 30 minutes (reading)

---

### Part 2: First CUDA Kernel
**File:** `02_first_kernel.cu`

**Topics Covered:**
- Writing your first `__global__` function
- Kernel launch syntax `<<<blocks, threads>>>`
- Memory allocation and transfers
- Error handling
- Vector addition example

**Code Example:**
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
```

**What You'll Build:**
- Complete vector addition program
- Error checking utilities
- Performance comparison with CPU

**Compile & Run:**
```bash
make 02_first_kernel
./02_first_kernel
```

**Expected Output:**
```
✓ Results verified successfully!
GPU time: 2.145 ms
CPU time: 45.632 ms
Speedup: 21.27x
```

**Time to Complete:** 1 hour

---

### Part 3: CUDA Memory Model
**File:** `03_memory_model.cu`

**Topics Covered:**
- Global, constant, and shared memory
- Memory coalescing
- Access patterns and performance
- Bank conflicts
- Bandwidth optimization

**Key Concepts:**
```
Memory Hierarchy (Speed → Size)
Fast ↑                      Small ↑
     Registers                   |
     Shared Memory               |
     L1/L2 Cache                 |
     Global Memory               |
Slow ↓                      Large ↓
```

**Performance Tests:**
- Coalesced vs strided access (5-10x difference)
- Global vs constant memory (2-3x speedup)
- Memory bandwidth analysis

**Compile & Run:**
```bash
make 03_memory_model
./03_memory_model
```

**Time to Complete:** 1.5 hours

---

### Part 4: Thread Organization
**File:** `04_thread_organization.cu`

**Topics Covered:**
- 1D, 2D, and 3D thread layouts
- Block and grid configuration
- Thread indexing formulas
- Optimal block sizes
- Occupancy considerations

**Visual Guide:**
```
2D Grid Example:
┌────┬────┬────┐
│B0,2│B1,2│B2,2│  Each block contains
├────┼────┼────┤  16x16 = 256 threads
│B0,1│B1,1│B2,1│
├────┼────┼────┤
│B0,0│B1,0│B2,0│
└────┴────┴────┘
```

**Applications:**
- Image processing (2D)
- Volume rendering (3D)
- Performance comparison with different block sizes

**Compile & Run:**
```bash
make 04_thread_organization
./04_thread_organization
```

**Time to Complete:** 1 hour

---

### Part 5: Matrix Operations
**File:** `05_matrix_operations.cu`

**Topics Covered:**
- Naive matrix multiplication
- Tiled matrix multiplication
- Matrix transpose optimization
- Performance comparison with cuBLAS
- Optimization strategies

**Implementations:**

1. **Naive Matrix Multiplication**
   - Simple but slow
   - ~50 GFLOPS

2. **Tiled with Shared Memory**
   - 3-5x faster
   - ~200 GFLOPS

3. **cuBLAS (Optimized)**
   - 10-20x faster than naive
   - ~1000+ GFLOPS

**Compile & Run:**
```bash
make 05_matrix_operations
./05_matrix_operations
```

**Expected Performance:**
```
┌─────────────────┬────────────┬────────────┐
│ Implementation  │ Time (ms)  │ GFLOPS     │
├─────────────────┼────────────┼────────────┤
│ Naive           │   245.123  │    52.34   │
│ Tiled           │    68.456  │   187.23   │
│ cuBLAS          │    12.789  │  1002.45   │
└─────────────────┴────────────┴────────────┘
```

**Time to Complete:** 2 hours

---

### Part 6: Shared Memory Optimization
**File:** `06_shared_memory.cu`

**Topics Covered:**
- Shared memory architecture
- Bank conflicts and how to avoid them
- Parallel reduction algorithms
- 1D stencil operations
- Prefix sum (scan)
- Histogram computation

**Key Techniques:**

1. **Reduction Optimization**
   - Naive: Baseline
   - Optimized: 2x faster (no divergence)
   - Unrolled: 3x faster (loop unrolling + warp ops)

2. **Bank Conflict Avoidance**
   ```c
   // Bad: 32-way bank conflict
   __shared__ float data[32][32];

   // Good: Padding eliminates conflicts
   __shared__ float data[32][33];
   ```

**Compile & Run:**
```bash
make 06_shared_memory
./06_shared_memory
```

**Time to Complete:** 2 hours

---

### Part 7: Thread Indexing Patterns (1D/2D/3D)
**File:** `11_thread_indexing_patterns.md`

**Topics Covered:**
- Understanding CUDA thread hierarchy
- 1D thread indexing for vectors and arrays
- 2D thread indexing for images and matrices
- 3D thread indexing for volumes and video
- Advanced patterns (checkerboard, tiled with halo)
- Common pitfalls and how to avoid them
- Performance considerations and memory coalescing

**Detailed Examples:**

1. **1D Patterns:**
   - Vector addition with global thread ID calculation
   - Strided access for grid-stride loops
   - Boundary checking best practices

2. **2D Patterns:**
   - Image grayscale conversion
   - Matrix transpose
   - Row-major vs column-major layout

3. **3D Patterns:**
   - Medical volume (CT/MRI) processing
   - Video temporal filtering
   - 3D stencil operations

4. **Advanced Techniques:**
   - Checkerboard access for iterative solvers
   - Tiled processing with halo regions
   - Avoiding bank conflicts

**Visual Highlights:**
```
2D Thread Organization:
┌────────────────────────────┐
│  Block Grid (3×2)          │
│  ┌──────┬──────┬──────┐    │
│  │(0,1) │(1,1) │(2,1) │    │
│  ├──────┼──────┼──────┤    │
│  │(0,0) │(1,0) │(2,0) │    │
│  └──────┴──────┴──────┘    │
│                            │
│  Each Block (4×4 threads): │
│  ┌─┬─┬─┬─┐                 │
│  │ │ │ │ │                 │
│  ├─┼─┼─┼─┤                 │
│  │ │ │ │ │                 │
│  └─┴─┴─┴─┘                 │
└────────────────────────────┘
```

**Common Issues Addressed:**
- Off-by-one errors in boundary checks
- X/Y axis confusion
- Row-major vs column-major confusion
- Uncoalesced memory access

**Who Should Read This:**
- Beginners learning thread indexing
- Developers working with images or volumes
- Anyone confused about 2D/3D indexing

**Prerequisites:**
- Complete Parts 1-4
- Basic understanding of arrays and matrices

**Read:** [11_thread_indexing_patterns.md](11_thread_indexing_patterns.md)

**Time to Complete:** 1.5 hours (reading + practice)

---

### Part 8: Streams and Async Operations
**File:** `07_streams_async.cu`

**Topics Covered:**
- CUDA streams for concurrency
- Asynchronous memory transfers
- Overlapping computation and communication
- Pinned memory
- Multi-GPU basics
- Stream synchronization

**Performance Pattern:**
```
WITHOUT STREAMS:
|H2D|Kernel|D2H|H2D|Kernel|D2H| = 60ms

WITH STREAMS:
Stream 0: |H2D|Kernel|D2H|
Stream 1:     |H2D|Kernel|D2H|
Stream 2:         |H2D|Kernel|D2H|
                              = 25ms (2.4x faster!)
```

**Compile & Run:**
```bash
make 07_streams_async
./07_streams_async
```

**Time to Complete:** 1.5 hours

---

### Part 8: Advanced Topics
**File:** `08_advanced_topics.cu`

**Topics Covered:**
- Atomic operations
- Warp-level primitives (shuffle, vote)
- Dynamic parallelism
- Unified Memory
- Cooperative groups
- Fast math intrinsics
- Performance optimization techniques

**Advanced Features:**

1. **Warp Shuffle** - Fast intra-warp communication
2. **Atomics** - Thread-safe operations
3. **Dynamic Parallelism** - Kernels launching kernels
4. **Unified Memory** - Simplified memory management

**Compile & Run:**
```bash
# Requires compute capability >= 3.5 for dynamic parallelism
make 08_advanced_topics
./08_advanced_topics
```

**Time to Complete:** 2.5 hours

---

### Part 9: Profiling and Debugging
**File:** `09_profiling_debugging.md`

**Topics Covered:**
- CUDA error checking best practices
- cuda-memcheck for memory errors
- cuda-gdb for debugging
- NVIDIA Nsight Systems (timeline profiling)
- NVIDIA Nsight Compute (kernel analysis)
- Performance metrics interpretation
- Common issues and solutions

**Essential Tools:**

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **cuda-memcheck** | Memory errors | Always before release |
| **cuda-gdb** | Interactive debugging | When kernel fails |
| **Nsight Systems** | System timeline | Find bottlenecks |
| **Nsight Compute** | Kernel optimization | Optimize hot spots |

**Example Workflow:**
```bash
# 1. Check for errors
cuda-memcheck ./program

# 2. Profile system-wide
nsys profile -o timeline ./program

# 3. Analyze critical kernel
ncu --set full -o analysis ./program
```

**Time to Complete:** 2 hours (reading + practice)

---

### Part 10: GPU Architecture Internals
**Files:** `10_gpu_architecture_internals.md`, `10_gpu_architecture_internals_part2.md`

**Topics Covered:**
- Complete GPU die architecture
- Streaming Multiprocessor (SM) deep dive
- Execution units and pipelines (CUDA cores, Tensor cores)
- Memory subsystem architecture
- Warp scheduling and execution
- Cache hierarchy (L1, L2, texture cache)
- Memory controllers (GDDR6X, HBM)
- Interconnect architecture (NVLink, PCIe)
- Architecture evolution (Kepler → Hopper)
- Performance characteristics and roofline model

**Detailed Coverage:**

**Part 1:**
- GPU die layout with GPCs, TPCs, and SMs
- SM microarchitecture (processing blocks, register files)
- FP32/INT32/FP64/Tensor Core pipelines
- Shared memory banking and bank conflicts
- Warp scheduler implementation
- Branch divergence handling

**Part 2:**
- L1/L2 cache structure and coherence
- Memory controller architecture
- GDDR vs HBM comparison
- NVLink protocol and topology
- PCIe interface details
- Generational architecture comparison
- Roofline performance model
- Latency and bandwidth numbers

**Visual Highlights:**
```
Streaming Multiprocessor Detail:
┌────────────────────────────────┐
│  Warp Schedulers (4)           │
├────────────────────────────────┤
│  Processing Blocks:            │
│  • 128 FP32 Cores              │
│  • 64 INT32 Cores              │
│  • 4 Tensor Cores              │
│  • Special Function Units      │
├────────────────────────────────┤
│  Register File (256 KB)        │
│  Shared Memory (128 KB)        │
└────────────────────────────────┘
```

**Who Should Read This:**
- Performance engineers
- Advanced CUDA developers
- Computer architecture enthusiasts
- Anyone optimizing at the hardware level

**Prerequisites:**
- Complete Parts 1-9
- Understanding of computer architecture basics
- Desire to understand hardware-level optimization

**Read:** [10_gpu_architecture_internals.md](10_gpu_architecture_internals.md)

**Time to Complete:** 3-4 hours (detailed reading)

---

### Part 12: Work Allocation & Execution Guide
**File:** `WORK_ALLOCATION_GUIDE.md`

**Topics Covered:**
- Complete hardware hierarchy (GPU → SMs → Warps → Cores)
- Software to hardware mapping (Grid → Blocks → Threads)
- Block size selection strategies (when to use 128, 256, 512, 1024)
- Grid size selection strategies (exact coverage, saturation, grid-stride)
- Occupancy optimization and limiting factors
- Work distribution patterns (element-wise, tiled, reduction, grid-stride)
- Real-world configuration examples
- Decision framework for choosing configurations

**Key Questions Answered:**
- When should I use 256 threads vs 512 threads per block?
- How many blocks should I launch?
- What's the relationship between occupancy and performance?
- How does work get mapped to CUDA cores and SMs?
- Which work distribution pattern should I use?

**Visual Highlights:**
```
Complete GPU Hierarchy:
GPU Device
└─ GPC × N
   └─ TPC × M
      └─ SM × 2
         ├─ Warp Schedulers × 4
         ├─ CUDA Cores × 128
         ├─ Registers (65,536)
         └─ Shared Memory (128 KB)
```

**Practical Decision Trees:**
- Block size selection based on workload type
- Grid size calculation formulas
- Resource usage analysis
- Performance trade-offs

**Real Examples with Analysis:**
```
Vector Addition (10M elements):
• Block: 256 threads
• Grid: 39,063 blocks
• Why: Simple compute, no shared memory
• Result: 82 SMs × 16 blocks each = full saturation ✓

Matrix Multiply (2048×2048):
• Block: 32×32 (1024 threads)
• Grid: 64×64 (4096 blocks)
• Why: 2D problem, heavy shared memory use
• Result: 100% occupancy, optimal reuse ✓
```

**Performance Benchmarks:**
Shows actual timing data for different block sizes:
- 32 threads: 2.1 ms (poor latency hiding)
- 256 threads: 0.5 ms (optimal) ✓
- 1024 threads: 0.6 ms (less flexible)

**Who Should Read This:**
- Anyone confused about block/grid sizing
- Developers wanting to optimize occupancy
- Engineers needing to understand work distribution
- Students learning CUDA execution model

**Prerequisites:**
- Complete Parts 1-4
- Understanding of thread hierarchy
- Basic knowledge of GPU architecture

**Read:** [WORK_ALLOCATION_GUIDE.md](WORK_ALLOCATION_GUIDE.md)

**Time to Complete:** 2-3 hours (comprehensive reading + practice)

---

## 🛠️ Setup Instructions

### System Requirements

**Hardware:**
- NVIDIA GPU with compute capability >= 7.0 (Volta or later recommended)
- Entry-level: RTX 2060 or GTX 1650 (Turing)
- Recommended: RTX 3060+ (Ampere) or RTX 4060+ (Ada)
- For Tensor Core examples: sm_70+ (Volta or later)
- For CUDA Tile examples: sm_80+ (Ampere or later)

**Software:**
- CUDA Toolkit 12.6 or later (13.3 recommended for latest features)
- Compatible C/C++ compiler:
  - Linux: GCC 7-14, Clang 11+
  - Windows: Visual Studio 2019+
- C++20 for CUDA Tile programming; C++23 supported in nvcc 13.3+
- CMake 3.18+ (optional, for build system)

### Installation

#### Linux (Ubuntu/Debian)

```bash
# 1. Check for NVIDIA GPU
lspci | grep -i nvidia

# 2. Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-XXX  # Replace XXX with version

# 3. Download CUDA Toolkit
# Visit: https://developer.nvidia.com/cuda-downloads
# Or use package manager:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# 4. Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Verify installation
nvcc --version
nvidia-smi
```

#### Windows

```cmd
1. Install Visual Studio 2019 or later
2. Download CUDA Toolkit installer from NVIDIA website
3. Run installer and follow prompts
4. Add to PATH:
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp
5. Verify: nvcc --version
```

#### macOS

**Note:** NVIDIA has discontinued CUDA support for macOS after 10.13.

For Apple Silicon Macs, consider Metal Performance Shaders or other alternatives.

---

## 🔨 Building Examples

### Using Make

```bash
# Build all examples
make all

# Build specific example
make 02_first_kernel

# Clean build artifacts
make clean

# Build with debug info
make debug

# Run all examples
make run
```

### Manual Compilation

```bash
# Basic compilation
nvcc -o program program.cu

# With optimization
nvcc -O3 -o program program.cu

# With debug info
nvcc -g -G -o program program.cu

# With specific architecture
nvcc -arch=sm_75 -o program program.cu

# With cuBLAS
nvcc -o program program.cu -lcublas

# With dynamic parallelism
nvcc -arch=sm_70 -rdc=true -o program program.cu -lcudadevrt
```

### Architecture Flags

| GPU Generation | Compute Capability | Flag | Example GPUs | Status |
|----------------|-------------------|------|--------------|--------|
| Maxwell | 5.0, 5.2 | `-arch=sm_52` | GTX 9xx | Feature-complete* |
| Pascal | 6.0, 6.1 | `-arch=sm_61` | GTX 10xx, P100 | Feature-complete* |
| Volta | 7.0 | `-arch=sm_70` | V100, Titan V | Feature-complete* |
| Turing | 7.5 | `-arch=sm_75` | RTX 20xx, T4 | Active |
| Ampere | 8.0, 8.6 | `-arch=sm_80` | A100, RTX 30xx | Active |
| Ada Lovelace | 8.9 | `-arch=sm_89` | RTX 40xx, L40 | Active |
| Hopper | 9.0 | `-arch=sm_90` | H100, H200 | Active |
| Blackwell DC | 10.0, 10.3 | `-arch=sm_100` | B200, GB200, GB300 | Active |
| Blackwell Consumer | 12.0, 12.1 | `-arch=sm_120` | RTX 50xx, DGX Spark | Active |

*Feature-complete: no new features; offline compilation will be removed in next major toolkit release.

Use the `a` suffix for architecture-accelerated features (e.g., `-arch=sm_90a` for
Hopper-specific features, `-arch=sm_100a` for Blackwell-specific features).

Check your GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## 📝 Quick Reference

### Common Patterns

#### Memory Management
```c
// Allocate
float *d_data;
cudaMalloc(&d_data, size);

// Copy to device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Copy to host
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Free
cudaFree(d_data);
```

#### Kernel Launch
```c
// 1D
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(args);

// 2D
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel<<<grid, block>>>(args);
```

#### Error Checking
```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

#### Thread Indexing
```c
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "nvcc: command not found"

**Solution:**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
# Or add to ~/.bashrc permanently
```

#### Issue: "cannot find -lcudart"

**Solution:**
```bash
# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Issue: "no CUDA-capable device is detected"

**Solutions:**
1. Check GPU: `lspci | grep -i nvidia`
2. Check driver: `nvidia-smi`
3. Reinstall driver if needed
4. Check if GPU is enabled in BIOS

#### Issue: "cudaErrorInvalidConfiguration"

**Common Causes:**
- Block size exceeds maximum (1024 threads)
- Grid size too large
- Shared memory exceeded

**Solution:**
```c
// Check device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max shared memory: %zu\n", prop.sharedMemPerBlock);
```

#### Issue: Slow Performance

**Debug Steps:**
1. Profile with Nsight Systems
2. Check occupancy
3. Verify memory coalescing
4. Look for bank conflicts
5. Measure bandwidth utilization

---

## 📚 Further Reading

### Official Documentation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

### Books

- **"Programming Massively Parallel Processors"** by Hwu, Kirk & Hajj (4th ed.)
  - The definitive textbook, updated through Hopper
  - Covers fundamentals to advanced topics

- **"CUDA by Example"** by Sanders & Kandrot
  - Great for beginners
  - Lots of practical examples

- **"Professional CUDA C Programming"** by Cheng, Grossman & McKercher
  - In-depth optimization techniques
  - Real-world applications

### Online Resources

- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) - Regular posts on new features and techniques
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) - The authoritative reference
- [CUDA Tile C++ API Reference](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/) - Modern tile programming
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) - High-performance GEMM templates and CuTe DSL
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) - Official code samples

### Libraries & Frameworks

- [CUTLASS](https://github.com/NVIDIA/cutlass) - High-performance linear algebra templates (supports FP4-FP64)
- [Triton](https://github.com/triton-lang/triton) - Python-based GPU kernel language
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) - Optimized BLAS on GPU
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) - Deep learning primitives
- [NCCL](https://docs.nvidia.com/deeplearning/nccl/) - Multi-GPU and multi-node collective communication
- [Thrust/CCCL](https://github.com/NVIDIA/cccl) - C++ parallel algorithms and containers

### Video Tutorials

- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [CUDA Training Series (ORNL)](https://www.olcf.ornl.gov/cuda-training-series/)
- [GTC On-Demand](https://www.nvidia.com/en-us/on-demand/) - Conference talks on latest GPU programming

### Community

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/)
- [Stack Overflow - CUDA Tag](https://stackoverflow.com/questions/tagged/cuda)
- [Reddit - r/CUDA](https://www.reddit.com/r/CUDA/)

---

## 🎯 Learning Path Recommendations

### Beginner (1-2 weeks)
1. Read Part 1 (Introduction)
2. Complete Part 2 (First Kernel)
3. Practice with simple examples
4. Learn error checking

### Intermediate (2-4 weeks)
1. Study Part 3 (Memory Model)
2. Complete Part 4 (Thread Organization)
3. Implement Part 5 (Matrix Operations)
4. Profile your code

### Advanced (4-8 weeks)
1. Master Part 6 (Shared Memory)
2. Learn Part 7 (Streams)
3. Explore Part 8 (Advanced Topics)
4. Study Part 9 (Profiling & Debugging)
5. Optimize real applications

### Expert (Ongoing)
- Read research papers
- Contribute to open-source projects
- Optimize for specific architectures
- Explore multi-GPU programming
- Learn CUDA libraries (cuDNN, cuFFT, etc.)

---

## 🏆 Practice Projects

### Beginner Projects
1. Vector operations (add, multiply, dot product)
2. Image filtering (blur, sharpen)
3. Histogram computation
4. Prime number finder

### Intermediate Projects
1. Conway's Game of Life
2. Mandelbrot set renderer
3. N-body simulation
4. K-means clustering

### Advanced Projects
1. Ray tracer
2. Neural network training
3. Fluid simulation
4. Molecular dynamics
5. Custom deep learning operators

---

## 📊 Performance Tips Summary

### Memory Optimization
✓ Coalesce global memory accesses
✓ Use shared memory for reused data
✓ Minimize host-device transfers
✓ Use pinned memory for async transfers
✓ Prefer texture memory for 2D spatial locality

### Compute Optimization
✓ Maximize occupancy (but not at all costs)
✓ Minimize branch divergence
✓ Use fast math when appropriate
✓ Unroll small loops
✓ Fuse operations to reduce memory traffic

### Execution Optimization
✓ Use streams for concurrency
✓ Overlap computation and communication
✓ Balance CPU and GPU work
✓ Consider multi-GPU for large problems
✓ Profile before optimizing

---

## 🤝 Contributing

Found an error? Have a suggestion? Want to add an example?

This tutorial is designed to be comprehensive and accurate. Feedback is welcome!

---

## 📜 License

This tutorial is provided for educational purposes. Code examples are free to use and modify.

CUDA® is a trademark of NVIDIA Corporation.

---

## 🙏 Acknowledgments

Based on:
- NVIDIA CUDA documentation
- Academic research papers
- Community best practices
- Years of GPU programming experience

---

## 📞 Getting Help

Stuck? Here's how to get help:

1. **Check the documentation** - Read the relevant tutorial section
2. **Review error messages** - Use cuda-memcheck and error checking
3. **Search online** - Stack Overflow, NVIDIA forums
4. **Simplify** - Create minimal reproducible example
5. **Profile** - Use Nsight tools to understand behavior

---

**Happy GPU Programming! 🚀**

Start with [Part 1: Introduction](01_introduction.md)

---

*Last Updated: May 2026*
*CUDA Version: 13.3 (also compatible with 12.x)*
*Tutorial Version: 2.0*

