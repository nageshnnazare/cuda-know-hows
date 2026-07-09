# CUDA Know-Hows — The One-Stop CUDA C++ Guide

> A single, deeply-documented path from *"what is a GPU?"* to *"I write
> tiled, warp-optimized kernels, overlap copies with compute across streams,
> and know when to reach for cuBLAS/CUTLASS instead."* Every chapter explains
> the **hardware reason** a technique works, shows **runnable CUDA C++**, and
> uses **ASCII diagrams** so you can *see* the grid, the warps, the memory, and
> the SMs. Nothing is hand-wavy: if we say something is fast, we say which
> hardware resource makes it fast and how to measure it.

Aligned with **CUDA Toolkit 13.x** and the modern CUDA Programming Guide
(unified memory, `cuda::ceil_div`, stream-ordered allocation, CUDA Graphs,
cooperative groups, Hopper/Blackwell features), while teaching the timeless
thread-centric fundamentals you need to understand *any* CUDA code.

---

## The mental model of GPU programming

A GPU is not "a faster CPU." It is a **throughput machine**: thousands of tiny
cores that hide memory latency by having enormous numbers of threads in flight.
You get speed not by making one thread fast, but by giving the machine so much
independent work that it never sits idle.

```
   CPU: few big cores, deep caches, branch prediction   -> LOW LATENCY per task
   GPU: 1000s of tiny cores, latency hidden by threads   -> HIGH THROUGHPUT

   ┌──────────── CPU ─────────────┐   ┌─────────────── GPU ──────────────────┐
   │ ┌────┐┌────┐  big cache      │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
   │ │core││core│ ┌───────────┐   │   │ ░░░ thousands of simple cores ░░░    │
   │ └────┘└────┘ │    L3     │   │   │ ░░░ grouped into SMs, fed by ░░░░    │
   │ ┌────┐┌────┐ └───────────┘   │   │ ░░░ 1000s of resident threads ░░     │
   │ │core││core│                 │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
   └──────────────────────────────┘   └──────────────────────────────────────┘
```

Everything in this guide flows from one question: **is the GPU busy, or is it
waiting?** Waiting on memory (uncoalesced access, no reuse), waiting on the host
(too many transfers), or waiting on itself (divergence, low occupancy). HPC on
the GPU is the art of removing the waiting.

---

## The guide map

```
              ┌───────────────────────────────────────────────────────┐
              │             CUDA KNOW-HOWS — LEARNING MAP             │
              └───────────────────────────────────────────────────────┘

  FOUNDATIONS                 MEMORY                    EXECUTION & PERF
  ┌──────────────────────┐    ┌──────────────────────┐  ┌──────────────────────┐
  │ 00 Introduction      │    │ 05 Memory model      │  │ 08 Execution & occ.  │
  │ 01 Setup & compile   │──▶ │ 06 Memory management │─▶│ 09 Work allocation   │
  │ 02 First kernel      │    │ 07 Shared memory     │  │ 10 GPU architecture  │
  │ 03 Thread hierarchy  │    │                      │  │ 11 Matrix multiply   │
  │ 04 Indexing patterns │    │                      │  │ 12 Atomics & sync    │
  └──────────────────────┘    └──────────────────────┘  └──────────────────────┘
             │                                                     │
             ▼                                                     ▼
  CONCURRENCY & ADVANCED                          PROFILING, LIBS & APPLICATIONS
  ┌──────────────────────┐                        ┌──────────────────────────────┐
  │ 13 Streams & concur. │                        │ 18 Profiling & debugging     │
  │ 14 Advanced kernels  │                        │ 19 Optimization case studies │
  │ 15 Advanced memory   │ ─────────────────────▶ │ 20 Libraries & ecosystem     │
  │ 16 CUDA graphs       │                        │ 21 Modern CUDA (Tile/FP8)    │
  │ 17 Multi-GPU         │                        │ 22 Applications              │
  └──────────────────────┘                        │ 99 Cheatsheet                │
                                                  └──────────────────────────────┘
```

## Table of contents

### Part I — Foundations
| # | Chapter | You'll master |
|---|---------|---------------|
| 00 | [Introduction](00_introduction.md) | CPU vs GPU, SIMT, thread/memory hierarchy, architecture generations, roofline |
| 01 | [Setup & compilation](01_setup_and_compilation.md) | Install, `nvcc` pipeline, architecture flags, PTX/SASS, the program lifecycle |
| 02 | [Your first kernel](02_first_kernel.md) | `__global__`, `<<<>>>`, unified vs explicit memory, error checking, vector add |
| 03 | [Thread hierarchy](03_thread_hierarchy.md) | grid/block/thread/warp, `dim3`, `ceil_div`, bounds checking, grid-stride loops |
| 04 | [Thread indexing patterns](04_thread_indexing_patterns.md) | 1D/2D/3D indexing for arrays, images, volumes; pitfalls & coalescing |

### Part II — Memory
| # | Chapter | You'll master |
|---|---------|---------------|
| 05 | [Memory model](05_memory_model.md) | Global/shared/constant/local/texture, **coalescing**, access patterns |
| 06 | [Memory management](06_memory_management.md) | `cudaMalloc`/managed/pinned/pooled, stream-ordered alloc, strategies |
| 07 | [Shared memory](07_shared_memory.md) | On-chip scratchpad, **bank conflicts**, reduction, scan, histogram, stencil |

### Part III — Execution & performance
| # | Chapter | You'll master |
|---|---------|---------------|
| 08 | [Execution model & occupancy](08_execution_model_and_occupancy.md) | Warp scheduling, divergence, latency hiding, occupancy |
| 09 | [Work allocation](09_work_allocation.md) | Choosing block/grid sizes, mapping work to SMs, decision framework |
| 10 | [GPU architecture](10_gpu_architecture.md) | SM microarchitecture, caches, memory controllers, NVLink, generations |
| 11 | [Matrix multiplication](11_matrix_multiplication.md) | Naive → tiled → cuBLAS; the canonical optimization journey |
| 12 | [Atomics & synchronization](12_atomics_and_synchronization.md) | Atomics, locks, `__syncthreads`, warp sync, cooperative groups |

### Part IV — Concurrency & advanced
| # | Chapter | You'll master |
|---|---------|---------------|
| 13 | [Streams & concurrency](13_streams_and_concurrency.md) | Streams, events, async copies, overlap, pinned memory |
| 14 | [Advanced kernel techniques](14_advanced_kernel_techniques.md) | Warp shuffle/vote, cooperative groups, dynamic parallelism, intrinsics |
| 15 | [Advanced memory techniques](15_advanced_memory_techniques.md) | Async copy (`cp.async`), TMA, pipelines, L2 residency, vectorized loads |
| 16 | [CUDA graphs](16_cuda_graphs.md) | Capture/instantiate/launch, cutting launch overhead, conditional nodes |
| 17 | [Multi-GPU](17_multi_gpu.md) | P2P, NVLink, NCCL collectives, scaling patterns |

### Part V — Profiling, libraries & applications
| # | Chapter | You'll master |
|---|---------|---------------|
| 18 | [Profiling & debugging](18_profiling_and_debugging.md) | Nsight Systems/Compute, `compute-sanitizer`, `cuda-gdb`, metrics, testing |
| 19 | [Optimization case studies](19_optimization_case_studies.md) | Worked before/after optimizations end-to-end |
| 20 | [Libraries & ecosystem](20_libraries_and_ecosystem.md) | cuBLAS/cuDNN/cuFFT/Thrust/CUB/CCCL/CUTLASS — don't reinvent |
| 21 | [Modern CUDA](21_modern_cuda.md) | CUDA Tile C++, FP8/FP4, thread-block clusters, Hopper/Blackwell |
| 22 | [Applications](22_applications.md) | Image processing, sorting, scientific, graphs, ML/DL — full examples |
| 99 | [Cheatsheet](99_cheatsheet.md) | Everything condensed: APIs, launch config, intrinsics, commands |

All runnable code lives in [`examples/`](examples/) with a
[`Makefile`](examples/Makefile) — see [`examples/README.md`](examples/README.md).

---

## How to use this guide

1. **Read 00–03 first.** They install the vocabulary (host/device, grid/block/
   warp, kernels, memory) that everything else assumes.
2. Read the rest **in order** the first time — memory (05–07) before performance
   (08–12) before advanced concurrency (13–17).
3. Keep [`99_cheatsheet.md`](99_cheatsheet.md) open while you code.
4. **Always measure** with Nsight before and after a change (Chapter 18).

## The golden rules of CUDA performance

```
   1. EXPOSE ENOUGH PARALLELISM.  The GPU hides latency with many warps. Give it
      far more threads than cores.
   2. COALESCE MEMORY ACCESS.     Thread i reads element i. Contiguous = 1 wide
      transaction; scattered = many. This is the #1 real-world win.
   3. REUSE DATA ON-CHIP.         Stage reused data in shared memory / registers
      (tiling). Turns memory-bound kernels compute-bound.
   4. MINIMIZE HOST<->DEVICE TRAFFIC.  PCIe is slow. Keep data resident; overlap
      the transfers you must do.
   5. AVOID WARP DIVERGENCE.      Keep all 32 lanes of a warp on the same path.
   6. USE THE LIBRARIES.          cuBLAS/cuDNN/CUTLASS beat hand-rolled kernels
      for standard ops. Hand-write only what they don't cover.
   7. PROFILE, DON'T GUESS.       Nsight tells you the real bottleneck.
```

## Prerequisites

- Comfortable C/C++ (pointers, memory, structs). For a refresher see the sibling
  `cpp_mastery` / `cpp-know-hows` guides; for CPU-side performance intuition
  (roofline, cache, SIMD, threads) see `cpp-hpc`, which pairs naturally with this.
- An NVIDIA GPU (compute capability ≥ 7.0 recommended) and CUDA Toolkit 12.6+
  (13.x recommended). The conceptual chapters are valuable even without a GPU.

Let's begin: [00 — Introduction →](00_introduction.md)

---

*CUDA® is a trademark of NVIDIA Corporation. This guide is for education; code is
free to use and modify.*
