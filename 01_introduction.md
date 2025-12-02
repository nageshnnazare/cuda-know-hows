# CUDA Tutorial - Part 1: Introduction to CUDA Programming

## Table of Contents
1. [What is CUDA?](#what-is-cuda)
2. [GPU Architecture](#gpu-architecture)
3. [Why Use GPUs?](#why-use-gpus)
4. [CUDA Programming Model](#cuda-programming-model)
5. [Setting Up CUDA](#setting-up-cuda)

---

## What is CUDA?

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables developers to use NVIDIA GPUs for general-purpose computing (GPGPU - General-Purpose computing on Graphics Processing Units).

### Key Concepts:
- **Host**: The CPU and its memory (host memory)
- **Device**: The GPU and its memory (device memory)
- **Kernel**: A function that runs on the GPU
- **Thread**: A single execution unit on the GPU

---

## GPU Architecture

### CPU vs GPU Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CPU ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │          │  │          │  │          │  │          │             │
│  │  Core 1  │  │  Core 2  │  │  Core 3  │  │  Core 4  │             │
│  │          │  │          │  │          │  │          │             │
│  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │             │
│  │ │Cache │ │  │ │Cache │ │  │ │Cache │ │  │ │Cache │ │             │
│  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │             │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
│                                                                     │
│                    ┌─────────────────────┐                          │
│                    │   Shared L3 Cache   │                          │
│                    └─────────────────────┘                          │
│                                                                     │
│  Features: Few cores, Complex control logic, Large caches           │
│  Best for: Sequential processing, Complex branching                 │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         GPU ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐   │
│  │   SM   │   SM   │   SM   │   SM   │   SM   │   SM   │   SM   │   │
│  │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │   │
│  ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ │   │
│  │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │   │
│  ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ │   │
│  │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │   │
│  │ Shared │ Shared │ Shared │ Shared │ Shared │ Shared │ Shared │   │
│  │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘   │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐   │
│  │   SM   │   SM   │   SM   │   SM   │   SM   │   SM   │   SM   │   │
│  │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │┌─┬─┬─┐ │   │
│  ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ │   │
│  │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │├─┼─┼─┤ │   │
│  ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ ││C│C│C│ │   │
│  │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │└─┴─┴─┘ │   │
│  │ Shared │ Shared │ Shared │ Shared │ Shared │ Shared │ Shared │   │
│  │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │  Mem   │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘   │
│                                                                     │
│  SM = Streaming Multiprocessor,  C = CUDA Core                      │
│  Features: Thousands of simple cores, Massive parallelism           │
│  Best for: Data-parallel operations, Simple computations            │
└─────────────────────────────────────────────────────────────────────┘
```

### GPU Components Explained:

1. **Streaming Multiprocessor (SM)**: 
   - The fundamental processing unit of a GPU
   - Contains multiple CUDA cores
   - Has its own shared memory and registers
   - Can execute one or more thread blocks

2. **CUDA Cores**: 
   - Basic computational units
   - Execute individual threads
   - Perform arithmetic and logical operations

3. **Memory Hierarchy** (from fastest to slowest):
   - **Registers**: Private to each thread, fastest access
   - **Shared Memory**: Shared within a block, very fast
   - **L1/L2 Cache**: Automatic caching
   - **Global Memory**: Accessible by all threads, slowest

---

## Why Use GPUs?

### Performance Comparison

```
Task: Vector Addition (1M elements)

CPU (Serial):           [████████████████████████████] 100ms
                        1 core processing sequentially

GPU (Parallel):         [██] 5ms
                        1000s of cores processing simultaneously

                        Speedup: 20x!
```

### Ideal GPU Workloads:

✅ **Good for GPU:**
- Large-scale data parallelism
- Simple, repetitive operations
- Matrix/vector operations
- Image/video processing
- Machine learning training
- Scientific simulations

❌ **Not ideal for GPU:**
- Sequential algorithms
- Heavy branching/conditionals
- Small data sets
- Complex control flow
- Frequent host-device transfers

---

## CUDA Programming Model

### The SIMT Architecture

CUDA uses **SIMT** (Single Instruction, Multiple Threads):
- Many threads execute the same instruction
- But on different data (SIMD-like)
- Each thread can follow its own path

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA THREAD HIERARCHY                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Grid (All threads in a kernel launch)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │                                                    │     │
│  │  Block (0,0)          Block (1,0)          Block (2,0)   │
│  │  ┌─────────────┐      ┌─────────────┐      ┌──────────┐  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  └─────────────┘      └─────────────┘      └──────────┘  │
│  │                                                          │
│  │  Block (0,1)          Block (1,1)          Block (2,1)   │
│  │  ┌─────────────┐      ┌─────────────┐      ┌──────────┐  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T│  │
│  │  └─────────────┘      └─────────────┘      └──────────┘  │
│  │                                                    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  T = Thread    Each thread has unique ID                    │
│  Threads grouped into Blocks                                │
│  Blocks grouped into Grid                                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Hierarchy Concepts:

1. **Thread**: 
   - Smallest execution unit
   - Has unique ID (threadIdx.x, threadIdx.y, threadIdx.z)
   - Executes kernel code

2. **Block**:
   - Group of threads (up to 1024 threads)
   - Threads in same block can cooperate via shared memory
   - Has unique ID (blockIdx.x, blockIdx.y, blockIdx.z)

3. **Grid**:
   - Collection of all blocks
   - One grid per kernel launch
   - Can be 1D, 2D, or 3D

### Memory Model

```
┌──────────────────────────────────────────────────────────────┐
│                      CUDA MEMORY HIERARCHY                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  HOST (CPU)                                                  │
│  ┌────────────────────────────────────────────┐              │
│  │         Host Memory (RAM)                  │              │
│  │         - Pageable Memory                  │              │
│  │         - Pinned Memory                    │              │
│  └────────────────────────────────────────────┘              │
│         ↕ ↕ ↕ (PCIe Transfer - Slow!)                        │
│  ┌────────────────────────────────────────────┐              │
│  │                                            │              │
│  │  DEVICE (GPU)                              │              │
│  │                                            │              │
│  │  ┌──────────────────────────────────────┐  │              │
│  │  │     Global Memory (Slow)             │  │              │
│  │  │     - Large (GBs)                    │  │              │
│  │  │     - Accessible by all threads      │  │              │
│  │  │     - Persistent across kernels      │  │              │
│  │  └──────────────────────────────────────┘  │              │
│  │              ↕                             │              │
│  │  ┌──────────────────────────────────────┐  │              │
│  │  │     L2 Cache (Automatic)             │  │              │
│  │  └──────────────────────────────────────┘  │              │
│  │              ↕                             │              │
│  │  ┌─────────────────┐  ┌─────────────────┐  │              │
│  │  │   SM 0          │  │   SM 1          │  │              │
│  │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │              │
│  │  │ │L1/Shared Mem│ │  │ │L1/Shared Mem│ │  │              │
│  │  │ │  (Fast)     │ │  │ │  (Fast)     │ │  │              │
│  │  │ └─────────────┘ │  │ └─────────────┘ │  │              │
│  │  │ ┌───┐┌───┐┌───┐ │  │ ┌───┐┌───┐┌───┐ │  │              │
│  │  │ │Reg││Reg││Reg│ │  │ │Reg││Reg││Reg│ │  │              │
│  │  │ └───┘└───┘└───┘ │  │ └───┘└───┘└───┘ │  │              │
│  │  │  (Fastest)      │  │  (Fastest)      │  │              │
│  │  └─────────────────┘  └─────────────────┘  │              │
│  │                                            │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  Speed:  Registers > Shared > L2 > Global > Host             │
│  Size:   Registers < Shared < L2 < Global < Host             │
└──────────────────────────────────────────────────────────────┘
```

### Memory Type Characteristics:

| Memory Type | Scope | Lifetime | Speed | Size |
|------------|-------|----------|-------|------|
| **Register** | Thread | Thread | Fastest | ~64KB per SM |
| **Local** | Thread | Thread | Slow (cached) | Per thread |
| **Shared** | Block | Block | Very Fast | ~48-96KB per SM |
| **Global** | Grid | Application | Slow | GBs |
| **Constant** | Grid | Application | Fast (cached) | 64KB |
| **Texture** | Grid | Application | Fast (cached) | Device dependent |

---

## Setting Up CUDA

### System Requirements:

1. **Hardware**: NVIDIA GPU with compute capability ≥ 3.0
2. **Software**: 
   - NVIDIA GPU drivers
   - CUDA Toolkit
   - C/C++ compiler (gcc/g++ or MSVC)

### Installation Steps (Linux):

```bash
# 1. Check GPU
lspci | grep -i nvidia

# 2. Download CUDA Toolkit from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# 3. Install (example for Ubuntu)
sudo apt-get update
sudo apt-get install nvidia-driver-XXX  # Latest driver
sudo apt-get install nvidia-cuda-toolkit

# 4. Verify installation
nvcc --version
nvidia-smi
```

### First CUDA Program Structure:

```
┌──────────────────────────────────────────────────────────┐
│              TYPICAL CUDA PROGRAM FLOW                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Initialize data on HOST (CPU)                        │
│     ↓                                                    │
│  2. Allocate memory on DEVICE (GPU)                      │
│     ↓                                                    │
│  3. Transfer data: HOST → DEVICE                         │
│     ↓                                                    │
│  4. Launch KERNEL (GPU computation)                      │
│     ↓                                                    │
│  5. Transfer results: DEVICE → HOST                      │
│     ↓                                                    │
│  6. Free GPU memory                                      │
│     ↓                                                    │
│  7. Process results on HOST                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### CUDA Function Qualifiers:

| Qualifier | Executed on | Callable from |
|-----------|-------------|---------------|
| `__global__` | Device (GPU) | Host (CPU) |
| `__device__` | Device (GPU) | Device (GPU) |
| `__host__` | Host (CPU) | Host (CPU) |

### Key CUDA API Functions:

```c
// Memory Management
cudaMalloc(void** devPtr, size_t size);           // Allocate GPU memory
cudaFree(void* devPtr);                           // Free GPU memory
cudaMemcpy(dst, src, size, direction);            // Copy memory

// Error Handling
cudaError_t err = cudaGetLastError();             // Get last error
const char* msg = cudaGetErrorString(err);        // Error to string

// Device Management
cudaGetDeviceCount(int* count);                   // Number of GPUs
cudaGetDeviceProperties(cudaDeviceProp*, device); // GPU properties
cudaSetDevice(int device);                        // Select GPU

// Synchronization
cudaDeviceSynchronize();                          // Wait for GPU to finish
```

---

## Performance Considerations

### Amdahl's Law and GPU Computing:

```
Speedup = 1 / ((1 - P) + P/S)

Where:
P = Portion of code that can be parallelized
S = Speedup of parallel portion

Example:
If 90% of code is parallelizable and GPU gives 100x speedup:
Speedup = 1 / (0.1 + 0.9/100) = 1 / 0.109 = 9.17x overall
```

### Optimization Goals:

1. **Maximize Parallelism**: Use all available threads
2. **Optimize Memory Access**: Coalesced reads/writes
3. **Minimize Transfers**: Reduce CPU↔GPU communication
4. **Utilize Shared Memory**: Reduce global memory access
5. **Avoid Divergence**: Minimize conditional branching

---

## Summary

In this introduction, we covered:

✓ What CUDA is and why it's useful  
✓ GPU architecture vs CPU architecture  
✓ CUDA thread hierarchy (Grid → Block → Thread)  
✓ Memory hierarchy and characteristics  
✓ Basic CUDA program flow  
✓ Setting up your CUDA environment  

### Next Steps:

In the next tutorial, we'll write our first CUDA kernel and understand:
- How to write `__global__` functions
- How to launch kernels with `<<<blocks, threads>>>`
- How to manage memory between host and device
- How to handle errors properly

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA QUICK REFERENCE                     │
├─────────────────────────────────────────────────────────────┤
│ Built-in Variables:                                         │
│   threadIdx.x/y/z    - Thread index within block            │
│   blockIdx.x/y/z     - Block index within grid              │
│   blockDim.x/y/z     - Block dimensions (threads per block) │
│   gridDim.x/y/z      - Grid dimensions (blocks per grid)    │
│                                                             │
│ Global Thread ID (1D):                                      │
│   int tid = blockIdx.x * blockDim.x + threadIdx.x;          │
│                                                             │
│ Kernel Launch:                                              │
│   kernel<<<gridSize, blockSize>>>(args);                    │
│                                                             │
│ Memory Transfer Directions:                                 │
│   cudaMemcpyHostToDevice    - CPU → GPU                     │
│   cudaMemcpyDeviceToHost    - GPU → CPU                     │
│   cudaMemcpyDeviceToDevice  - GPU → GPU                     │
│                                                             │
│ Synchronization:                                            │
│   cudaDeviceSynchronize()   - Wait for GPU                  │
│   __syncthreads()           - Wait within block (device)    │
└─────────────────────────────────────────────────────────────┘
```

Ready to write your first CUDA kernel? Continue to **02_first_kernel.cu**!

