# CUDA Tutorial - Part 1: Introduction to CUDA Programming

## Table of Contents
1. [What is CUDA?](#what-is-cuda)
2. [GPU Architecture](#gpu-architecture)
3. [Why Use GPUs?](#why-use-gpus)
4. [CUDA Programming Model](#cuda-programming-model)
5. [Architecture Generations](#architecture-generations)
6. [Setting Up CUDA](#setting-up-cuda)

---

## What is CUDA?

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables developers to use NVIDIA GPUs for general-purpose computing (GPGPU - General-Purpose computing on Graphics Processing Units).

As of 2026, CUDA has matured through major toolkit releases up to **CUDA Toolkit 13.3**, which introduces CUDA Tile C++ programming, C++23 support, and the CompileIQ auto-tuning framework. The platform now spans from consumer GeForce GPUs to datacenter-scale systems like the GB200 NVL72 with 72 interconnected Blackwell GPUs.

### Key Concepts:
- **Host**: The CPU and its memory (host memory)
- **Device**: The GPU and its memory (device memory)
- **Kernel**: A function that runs on the GPU, launched from the host
- **Thread**: A single execution unit on the GPU
- **Warp**: A group of 32 threads that execute in lockstep (the fundamental scheduling unit)
- **Tile** (CUDA 13.1+): A higher-level abstraction over threads that automatically maps to tensor cores and shared memory

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
│   ┌──────────────────────┐    ┌──────────────────────┐              │
│   │   Shared L2 Cache    │    │   Shared L2 Cache    │              │
│   └──────────────────────┘    └──────────────────────┘              │
│    ┌─────────────────────────────────────────────────┐              │
│    │                 Shared L3 Cache                 │              │
│    └─────────────────────────────────────────────────┘              │
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
   - Contains multiple CUDA cores, Tensor Cores, and special function units
   - Has its own shared memory, register file, and warp schedulers
   - Can execute one or more thread blocks concurrently
   - Modern SMs (Blackwell): 128 FP32 cores, 4 Tensor Cores, 4 warp schedulers per SM

2. **CUDA Cores**: 
   - Basic computational units for scalar FP32/INT32 operations
   - Execute individual threads within a warp
   - Modern GPUs: thousands of CUDA cores (e.g., B200 has ~18,000+ FP32 cores)

3. **Tensor Cores** (Volta and later):
   - Specialized matrix multiply-accumulate (MMA) units
   - Accelerate mixed-precision matrix operations (FP16, BF16, TF32, FP8, FP4)
   - Critical for deep learning and HPC workloads
   - 5th-gen Tensor Cores (Blackwell) support native FP4/FP6 with hardware rescaling

4. **Memory Hierarchy** (from fastest to slowest):
   - **Registers**: Private to each thread, fastest access (~1 cycle)
   - **Tensor Memory (TMEM)**: Blackwell-only, dedicated on-chip memory for tensor data
   - **Shared Memory**: Shared within a block, very fast (~5 cycles)
   - **L1/L2 Cache**: Automatic caching
   - **Global Memory (HBM)**: Accessible by all threads, high bandwidth but high latency (~400 cycles)

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
- A warp of 32 threads fetches the same instruction
- Each thread operates on different data (SIMD-like, but more flexible)
- Threads can diverge (follow different branches), but divergent paths serialize within a warp
- Since Volta (2017), threads within a warp have independent program counters,
  enabling fine-grained divergence and reconvergence

```
┌──────────────────────────────────────────────────────────────────┐
│                    CUDA THREAD HIERARCHY                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Grid (All threads in a kernel launch)                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Block (0,0)          Block (1,0)           Block (2,0)     | │
│  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  | │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  | │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  | │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  │ │
│  │  └─────────────┘      └─────────────┘      └─────────────┘  │ │
│  │                                                             │ │
│  │  Block (0,1)          Block (1,1)          Block (2,1)      │ │
│  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │ │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  │ │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  │ │
│  │  │ T T T T T T │      │ T T T T T T │      │ T T T T T T │  │ │
│  │  └─────────────┘      └─────────────┘      └─────────────┘  │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  T = Thread    Each thread has unique ID                         │
│  Threads grouped into Blocks                                     │
│  Blocks grouped into Grid                                        │
└──────────────────────────────────────────────────────────────────┘
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

| Memory Type | Scope | Lifetime | Latency | Size | Notes |
|------------|-------|----------|---------|------|-------|
| **Register** | Thread | Thread | ~1 cycle | ~256KB per SM | Fastest, most precious resource |
| **TMEM** | SM | Block | ~1 cycle | Per SM | Blackwell only, for tensor data |
| **Local** | Thread | Thread | ~400 cycles | Per thread | Spills to global when registers exhausted |
| **Shared** | Block | Block | ~5 cycles | 48-228KB per SM | Configurable L1/shared split |
| **L2 Cache** | Device | Auto | ~30 cycles | 6-96MB | Ampere+ allows persistence control |
| **Global** | Grid | Application | ~400 cycles | Up to 192GB (HBM3e) | Main GPU memory |
| **Constant** | Grid | Application | ~5 cycles | 64KB | Broadcast-optimized cache |
| **Texture** | Grid | Application | ~5 cycles | Device dependent | Spatial locality cache, filtering HW |

---

## Architecture Generations

Understanding GPU architecture generations is essential for writing portable,
high-performance code. Each generation introduces new capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GPU ARCHITECTURE EVOLUTION                        │
├────────────┬────────┬───────────────────────────────────────────────────────┤
│ Generation │ SM     │ Key Innovations                                       │
├────────────┼────────┼───────────────────────────────────────────────────────┤
│ Kepler     │ sm_35  │ Dynamic parallelism, Hyper-Q                          │
│ Maxwell    │ sm_52  │ Improved power efficiency, shared memory redesign     │
│ Pascal     │ sm_61  │ Unified Memory improvements, NVLink 1.0, HBM2         │
│ Volta      │ sm_70  │ Tensor Cores (1st gen), independent thread scheduling │
│ Turing     │ sm_75  │ RT Cores, INT8/INT4 Tensor Cores, mixed precision     │
│ Ampere     │ sm_80  │ TF32, BF16, 3rd-gen Tensor Cores, async copy          │
│ Ada        │ sm_89  │ FP8 Tensor Cores, Shader Execution Reordering         │
│ Hopper     │ sm_90  │ TMA, DPX, FP8, Thread Block Clusters, WGMMA           │
│ Blackwell  │ sm_100 │ FP4/FP6, 5th-gen Tensor Cores, Tensor Memory (TMEM)   │
│ Blackwell  │ sm_120 │ Consumer RTX 50-series, block-scaled MMA              │
│   (consumer)│       │ (no tcgen05, uses HMMA/WGMMA paths)                   │
└────────────┴────────┴───────────────────────────────────────────────────────┘

Feature-Complete (no new features in future CUDA):
  Maxwell (sm_52), Pascal (sm_61), Volta (sm_70)
  Offline compilation will be removed in the next major CUDA release.
```

### Blackwell Architecture Highlights (2025-2026):
- **208 billion transistors** on a dual-die design (two reticle-limited dies
  connected by a 10 TB/s chip-to-chip interconnect)
- **5th-gen Tensor Cores**: native FP4 and FP6 with hardware rescaling,
  single-thread MMA via `tcgen05.mma` (replaces warp-synchronous MMA)
- **Tensor Memory (TMEM)**: dedicated on-chip memory for tensor operands,
  reducing reliance on shared memory and register files during matrix ops
- **CTA Pair Execution**: two CTAs share operands through an intra-TPC network
- **192 GB HBM3e** at 8 TB/s bandwidth (B200)
- **NVLink 5**: 1.8 TB/s per GPU (up from 900 GB/s on Hopper)

### Hopper Architecture Highlights (2022-2024):
- **Thread Block Clusters**: new hierarchy level, groups of blocks that can
  cooperate via distributed shared memory
- **Tensor Memory Accelerator (TMA)**: hardware unit for async bulk data
  movement, offloading address computation from CUDA cores
- **WGMMA**: warp-group matrix multiply-accumulate (groups of 4 warps = 128 threads)
- **DPX Instructions**: hardware-accelerated dynamic programming
- **FP8** (e4m3, e5m2): native 8-bit floating-point for AI training/inference

---

## Setting Up CUDA

### System Requirements:

1. **Hardware**: NVIDIA GPU with compute capability >= 7.0 recommended
   - Minimum: sm_70 (Volta) for full modern CUDA feature support
   - Note: Maxwell/Pascal/Volta are feature-complete and will lose offline
     compilation support in the next major toolkit release
2. **Software**: 
   - NVIDIA GPU drivers (R550+ for CUDA 12.x, R580+ for CUDA 13.x)
   - CUDA Toolkit (13.3 is the latest as of May 2026)
   - C/C++ compiler: GCC 7-14, Clang 11+, or MSVC 2019+
   - C++20 required for CUDA Tile programming; C++23 officially supported in nvcc

### Installation Steps (Linux):

```bash
# 1. Check GPU
lspci | grep -i nvidia

# 2. Download CUDA Toolkit from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# 3. Install (example for Ubuntu 22.04/24.04)
sudo apt-get update
sudo apt-get install nvidia-driver-565  # Or latest for your GPU
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
cudaMemcpy(dst, src, size, direction);            // Copy memory (synchronous)
cudaMemcpyAsync(dst, src, size, dir, stream);     // Copy memory (async)
cudaMallocManaged(void** devPtr, size_t size);    // Unified Memory allocation

// Modern Memory Management (CUDA 11.2+)
cudaMallocAsync(void** devPtr, size_t size, stream);  // Stream-ordered alloc
cudaFreeAsync(void* devPtr, stream);                  // Stream-ordered free
cudaMemPool_t pool;                                   // Memory pool handles

// Error Handling
cudaError_t err = cudaGetLastError();             // Get last error
const char* msg = cudaGetErrorString(err);        // Error to string

// Device Management
cudaGetDeviceCount(int* count);                   // Number of GPUs
cudaGetDeviceProperties(cudaDeviceProp*, device); // GPU properties
cudaSetDevice(int device);                        // Select GPU

// Synchronization
cudaDeviceSynchronize();                          // Wait for GPU to finish

// CUDA Graphs (CUDA 10+, enhanced in 12.8+ with IF/ELSE/SWITCH nodes)
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStreamBeginCapture(stream, mode);             // Begin graph capture
cudaStreamEndCapture(stream, &graph);             // End capture
cudaGraphInstantiate(&graphExec, graph, 0);       // Compile graph
cudaGraphLaunch(graphExec, stream);               // Execute graph
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

### The Roofline Model

The roofline model helps determine whether your kernel is compute-bound
or memory-bound:

```
Arithmetic Intensity = FLOPs / Bytes Transferred

  Performance │
  (GFLOPS)    │          ╱─────────── Peak Compute (ceiling)
              │        ╱
              │      ╱
              │    ╱  ← Memory bandwidth wall
              │  ╱
              │╱
              └──────────────────────
                Arithmetic Intensity (FLOP/byte)

If your kernel is below the roofline:
  - Left of the ridge: memory-bound → optimize data access
  - Right of the ridge: compute-bound → optimize arithmetic
```

### Optimization Goals:

1. **Maximize Parallelism**: Saturate the GPU with enough concurrent work
2. **Optimize Memory Access**: Coalesced reads/writes, minimize transactions
3. **Minimize Transfers**: Reduce CPU↔GPU communication; use pinned/unified memory
4. **Utilize Shared Memory**: Stage data in fast on-chip memory
5. **Avoid Divergence**: Minimize conditional branching within warps
6. **Use Appropriate Precision**: FP16/BF16/TF32/FP8 via Tensor Cores when applicable
7. **Leverage CUDA Graphs**: Reduce launch overhead for repeated kernel sequences
8. **Profile Before Optimizing**: Use Nsight Systems and Nsight Compute to find real bottlenecks

---

## Programming Model Evolution: Threads to Tiles

CUDA's programming model has evolved significantly:

```
2007-2022: Thread-Centric Programming
  - Developer manually manages individual threads
  - Explicit shared memory, synchronization, indexing
  - Full control, but high complexity for advanced features
  - This is what most of this tutorial teaches (and is still essential)

2023 (Hopper): Warp-Group and Cluster Programming
  - WGMMA: 128 threads (4 warps) cooperate on matrix operations
  - Thread Block Clusters: blocks cooperate via distributed shared memory
  - TMA: hardware-managed async data movement

2024-2026 (CUDA 13.x): Tile-Based Programming
  - CUDA Tile: developer thinks in data tiles, not individual threads
  - Compiler automatically maps tiles to tensor cores, shared memory, TMA
  - Works across architectures (Ampere, Ada, Hopper, Blackwell)
  - Higher abstraction = easier to write, but thread-level knowledge
    is still essential for understanding performance
```

Most of this tutorial teaches thread-centric programming because it is the
foundation. You need to understand threads, warps, blocks, and memory to
effectively use (and debug) higher-level abstractions like CUDA Tile.

---

## Summary

In this introduction, we covered:

- What CUDA is and why it's useful (including CUDA 13.3 and the current ecosystem)
- GPU architecture vs CPU architecture
- CUDA thread hierarchy (Grid → Block → Warp → Thread)
- Memory hierarchy and characteristics (including modern additions like TMEM)
- Architecture generations from Kepler through Blackwell
- The roofline model for understanding performance limits
- Evolution from thread-centric to tile-based programming
- Basic CUDA program flow
- Setting up your CUDA environment

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

### Recommended Further Reading:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (official, always up to date)
- [CUDA Toolkit 13.3 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA Tile C++ API Reference](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/)
- [Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)

Ready to write your first CUDA kernel? Continue to [**02_first_kernel.cu**](./02_first_kernel.cu)!

