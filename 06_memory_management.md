# 06 — Memory Management

> Part of **[CUDA Know-Hows](README.md)**. Prev: [05 — Memory model](05_memory_model.md).
> Next: [07 — Shared memory](07_shared_memory.md). The complete reference to
> allocating and moving memory: `cudaMalloc`, unified, pinned, pooled/stream-
> ordered allocation, and choosing the right strategy.

## Host Memory, Device Memory, and Everything In Between

This guide is a deep-dive reference for every memory allocation strategy in CUDA.
It covers the *why*, the *when*, the *how*, and the performance implications of each
approach, from basic `malloc` to Blackwell's Tensor Memory.

Last Updated: May 2026 | CUDA Toolkit 13.3

---

## Table of Contents

1. [Memory Landscape Overview](#1-memory-landscape-overview)
2. [Host-Side Memory Allocation](#2-host-side-memory-allocation)
   - [Pageable Memory (malloc)](#21-pageable-memory-malloc)
   - [Pinned Memory (cudaHostAlloc)](#22-pinned-memory-cudahostalloc)
   - [Write-Combined Memory](#23-write-combined-memory)
   - [Registered Memory (cudaHostRegister)](#24-registered-memory-cudahostregister)
3. [Device-Side Memory Allocation](#3-device-side-memory-allocation)
   - [Global Memory (cudaMalloc)](#31-global-memory-cudamalloc)
   - [Shared Memory (__shared__)](#32-shared-memory-__shared__)
   - [Constant Memory (__constant__)](#33-constant-memory-__constant__)
   - [Texture Memory](#34-texture-memory)
   - [Registers](#35-registers)
   - [Local Memory](#36-local-memory)
4. [Unified / Managed Memory](#4-unified--managed-memory)
5. [Stream-Ordered Allocation (cudaMallocAsync)](#5-stream-ordered-allocation)
6. [Zero-Copy Memory](#6-zero-copy-memory)
7. [L2 Cache Persistence (Ampere+)](#7-l2-cache-persistence)
8. [Tensor Memory - TMEM (Blackwell)](#8-tensor-memory---tmem-blackwell)
9. [Memory Transfer Patterns](#9-memory-transfer-patterns)
10. [Decision Framework](#10-decision-framework)
11. [Common Pitfalls](#11-common-pitfalls)
12. [API Quick Reference](#12-api-quick-reference)

---

## 1. Memory Landscape Overview

Every CUDA program deals with two physical memory systems connected by
an interconnect (PCIe or NVLink):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CUDA MEMORY LANDSCAPE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HOST (CPU Side)                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │      │
│  │  │   Pageable   │  │    Pinned    │  │     Managed / Unified    │ │      │
│  │  │   (malloc)   │  │ (cudaHost    │  │   (cudaMallocManaged)    │ │      │
│  │  │              │  │    Alloc)    │  │                          │ │      │
│  │  │ Can be paged │  │ Page-locked  │  │  Visible to both CPU     │ │      │
│  │  │ to disk by   │  │ in physical  │  │  and GPU. Runtime        │ │      │
│  │  │ the OS. Must │  │ RAM. DMA-    │  │  auto-migrates pages     │ │      │
│  │  │ be staged    │  │ accessible   │  │  on demand.              │ │      │
│  │  │ through a    │  │ directly.    │  │                          │ │      │
│  │  │ pinned buffer│  │              │  │                          │ │      │
│  │  │ before DMA.  │  │              │  │                          │ │      │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│         │                    │                       │                      │
│         │ (slow: staging)    │ (fast: direct DMA)    │ (auto-migration)     │
│         ▼                    ▼                       ▼                      │
│  ═══════════════════════ PCIe Gen5 / NVLink 5 ═══════════════════════       │
│         │                    │                       │                      │
│         ▼                    ▼                       ▼                      │
│  DEVICE (GPU Side)                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                                                                    │     │
│  │  ┌──────────────────────────────────────────────────────────┐      │     │
│  │  │  GLOBAL MEMORY (HBM3e / GDDR6X)                          │      │     │
│  │  │  Largest, slowest GPU memory. Persists across kernels.   │      │     │
│  │  │  Allocated with cudaMalloc / cudaMallocAsync.            │      │     │
│  │  │  Access latency: ~400-800 cycles                         │      │     │
│  │  │  Bandwidth: 1-8 TB/s depending on GPU                    │      │     │
│  │  └─────────────────────────┬────────────────────────────────┘      │     │
│  │                            │                                       │     │
│  │  ┌─────────────────────────▼────────────────────────────────┐      │     │
│  │  │  L2 CACHE (6-96 MB, automatic)                           │      │     │
│  │  │  Caches global memory accesses. Ampere+ supports         │      │     │
│  │  │  persistence control (reserve portion for hot data).     │      │     │
│  │  └─────────────────────────┬────────────────────────────────┘      │     │
│  │                            │                                       │     │
│  │  Per-SM Resources:         │                                       │     │
│  │  ┌─────────────────────────▼────────────────────────────────┐      │     │
│  │  │  L1 CACHE / SHARED MEMORY (48-228 KB per SM)             │      │     │
│  │  │  Configurable split. Shared memory is programmer-managed.│      │     │
│  │  │  Access latency: ~5 cycles (shared), ~30 cycles (L1)     │      │     │
│  │  ├──────────────────────────────────────────────────────────┤      │     │
│  │  │  CONSTANT CACHE (8 KB per SM, backs 64 KB __constant__)  │      │     │
│  │  │  Optimized for broadcast: all threads reading same addr. │      │     │
│  │  ├──────────────────────────────────────────────────────────┤      │     │
│  │  │  TEXTURE CACHE (per SM)                                  │      │     │
│  │  │  Optimized for 2D spatial locality. Hardware filtering.  │      │     │
│  │  ├──────────────────────────────────────────────────────────┤      │     │
│  │  │  REGISTER FILE (256 KB per SM, 65536 × 32-bit)           │      │     │
│  │  │  Fastest storage. Private to each thread. ~1 cycle.      │      │     │
│  │  ├──────────────────────────────────────────────────────────┤      │     │
│  │  │  TENSOR MEMORY / TMEM (Blackwell sm_100 only)            │      │     │
│  │  │  Dedicated on-chip memory for tensor core operands.      │      │     │
│  │  │  Explicitly managed via tcgen05 PTX instructions.        │      │     │
│  │  └──────────────────────────────────────────────────────────┘      │     │
│  │                                                                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### At-a-Glance Comparison

| Memory | Location | Scope | Latency | Size | Read/Write | Allocated By |
|--------|----------|-------|---------|------|------------|--------------|
| Register | GPU chip | Thread | ~1 cycle | 256 KB/SM | R/W | Compiler |
| TMEM | GPU chip | SM | ~1 cycle | Per SM | R/W | PTX (tcgen05) |
| Local | Off-chip* | Thread | ~400 cyc | Per thread | R/W | Compiler (spill) |
| Shared | GPU chip | Block | ~5 cyc | 48-228 KB/SM | R/W | `__shared__` |
| Constant | Off-chip+cache | Grid | ~5 cyc** | 64 KB | R only | `__constant__` |
| Texture | Off-chip+cache | Grid | ~5 cyc** | Global | R only*** | Texture API |
| Global | Off-chip (HBM) | Grid+Host | ~400 cyc | Up to 192 GB | R/W | `cudaMalloc` |
| Pinned | Host RAM | Host+Device | PCIe/NVLink | Host RAM | R/W | `cudaHostAlloc` |
| Managed | Both | Host+Device | Varies | Both | R/W | `cudaMallocManaged` |

*Local memory resides in global memory but is cached in L1/L2.  
**When cached; uncached access is ~400 cycles.  
***Surface objects allow writes to texture-backed memory.

---

## 2. Host-Side Memory Allocation

### 2.1 Pageable Memory (malloc)

Standard C/C++ heap allocation. The OS virtual memory manager can swap
these pages to disk at any time.

```c
float *h_data = (float*)malloc(N * sizeof(float));
// ... use on CPU ...
free(h_data);
```

**How transfers work with pageable memory:**

When you call `cudaMemcpy` with pageable source memory, the CUDA driver must:

```
Step 1: Allocate a temporary pinned staging buffer
Step 2: Copy data from pageable → staging buffer (CPU memcpy)
Step 3: DMA transfer from staging buffer → GPU memory
Step 4: Free staging buffer

Total: 2 copies instead of 1
```

**Characteristics:**
- Simplest to use (standard C/C++)
- Slowest for host↔device transfers (requires internal staging)
- Can be swapped to disk by OS (pages may not be in RAM when DMA starts)
- No limit beyond available virtual memory
- Cannot be used for async transfers (`cudaMemcpyAsync` falls back to sync)

**When to use:**
- Prototyping and non-performance-critical code
- When allocation size is unpredictable
- When you don't want to consume physical RAM exclusively

---

### 2.2 Pinned Memory (cudaHostAlloc)

Page-locked (pinned) memory is locked into physical RAM and cannot be swapped
to disk. The DMA engine can transfer directly without staging.

```c
float *h_pinned;
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocDefault);
// ... use on CPU, transfer to GPU ...
cudaFreeHost(h_pinned);
```

**How transfers work with pinned memory:**

```
Step 1: DMA transfer directly from pinned buffer → GPU memory

Total: 1 copy (direct DMA)
```

**Transfer speed comparison:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                  HOST → DEVICE TRANSFER BANDWIDTH                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Pageable:  ████████████░░░░░░░░░░░░░░░░░░░░  ~6-8 GB/s  (PCIe 4)    │
│  Pinned:    ████████████████████████████░░░░░  ~24-26 GB/s (PCIe 4)  │
│  Pinned:    ████████████████████████████████░  ~50+ GB/s  (PCIe 5)   │ 
│                                                                      │
│  Speedup: 3-4x with pinned memory                                    │
│                                                                      │
│  NVLink (Grace-Blackwell):                                           │
│  Pinned:    ████████████████████████████████   ~450 GB/s  (NVLink)   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Flags for cudaHostAlloc:**

| Flag | Effect |
|------|--------|
| `cudaHostAllocDefault` | Standard pinned memory. Fast transfers. |
| `cudaHostAllocPortable` | Usable with any CUDA context (multi-GPU). |
| `cudaHostAllocMapped` | Maps into device address space (zero-copy). GPU can access directly. |
| `cudaHostAllocWriteCombined` | Write-combined: faster CPU writes, slower CPU reads. Best for write-only staging. |

**Characteristics:**
- 3-4x faster transfers vs pageable (direct DMA, no staging)
- Enables true asynchronous transfers with `cudaMemcpyAsync`
- Enables overlapping transfers and compute (requires streams)
- Reduces physical RAM available to the OS (pinned = not swappable)
- Allocating too much can cause OS instability or OOM
- Allocation itself is slower than `malloc` (OS must find contiguous physical pages)

**When to use:**
- High-throughput data pipelines (streaming data to GPU)
- When overlapping H2D transfer and kernel execution
- Double/triple buffering patterns
- Frequent small transfers where latency matters

**Rule of thumb:** Pin only what you need. Don't pin your entire dataset.

---

### 2.3 Write-Combined Memory

A variant of pinned memory optimized for unidirectional CPU→GPU data flow.

```c
float *h_wc;
cudaHostAlloc(&h_wc, size,
              cudaHostAllocMapped | cudaHostAllocWriteCombined);
```

**How it works:**

Normal pinned memory uses CPU write-back caching: the CPU caches writes in
L1/L2 and flushes them later. Write-combined memory bypasses CPU caches
entirely and accumulates writes in a write-combining buffer before flushing
to memory in burst.

```
Write-Back (normal):           Write-Combined:
CPU → L1 → L2 → RAM            CPU → WC buffer → RAM (burst)
                                      (bypasses L1/L2)

CPU Write speed:  Normal        CPU Write speed:  Faster
CPU Read speed:   Normal        CPU Read speed:   Very Slow (~10x slower)
GPU DMA Read:     Normal        GPU DMA Read:     Faster (no snoop needed)
```

**When to use:**
- CPU produces data, GPU consumes it (one-way flow)
- Streaming video frames to GPU
- Never when CPU needs to read back from this buffer

**When NOT to use:**
- CPU needs to read from this memory (extremely slow reads)
- Bidirectional data sharing

---

### 2.4 Registered Memory (cudaHostRegister)

Convert already-allocated pageable memory into pinned memory at runtime.
Useful when memory is allocated by a library or mmap'd file.

```c
float *h_data = (float*)malloc(size);
cudaHostRegister(h_data, size, cudaHostRegisterDefault);
// ... now behaves like pinned memory for transfers ...
cudaHostUnregister(h_data);
free(h_data);
```

**Flags:**

| Flag | Effect |
|------|--------|
| `cudaHostRegisterDefault` | Standard registration |
| `cudaHostRegisterPortable` | Usable across CUDA contexts |
| `cudaHostRegisterMapped` | Also maps to device address space |
| `cudaHostRegisterIoMemory` | For memory-mapped I/O regions |

**When to use:**
- Interfacing with libraries that allocate their own buffers
- Memory-mapped files (mmap)
- When refactoring existing code to use pinned transfers

---

## 3. Device-Side Memory Allocation

### 3.1 Global Memory (cudaMalloc)

The main GPU memory backed by HBM (datacenter) or GDDR (consumer).
Accessible by all threads across all blocks and persists across kernel launches.

```c
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));
// ... launch kernels that use d_data ...
cudaFree(d_data);
```

**Key characteristics:**
- Largest memory on GPU (up to 192 GB on B200, 288 GB on Blackwell Ultra)
- High bandwidth (1-8 TB/s depending on GPU)
- High latency (~400-800 cycles uncached)
- Cached in L2 (all architectures) and L1 (Volta+)
- Coalesced access is critical for performance

**Coalescing rules (what every CUDA programmer must know):**

```
COALESCED (good):                  STRIDED (bad):
Warp threads access consecutive    Warp threads access with gaps
memory → 1 memory transaction      → multiple transactions

Thread:  T0  T1  T2  T3  ...      Thread:  T0  T1  T2  T3  ...
Address: [0] [1] [2] [3]  ...     Address: [0] [4] [8] [12] ...
         └──── 128 bytes ────┘              └── scattered ──────┘
         = 1 transaction                    = up to 32 transactions
```

**Alignment:** `cudaMalloc` always returns 256-byte aligned pointers.
Accessing from a non-aligned offset (e.g., `ptr + 1`) may cost an extra
transaction. Structure padding can help.

**Sizing consideration:** GPU memory is a limited, shared resource.
Check available memory before large allocations:

```c
size_t free_bytes, total_bytes;
cudaMemGetInfo(&free_bytes, &total_bytes);
printf("Free: %.2f GB / %.2f GB\n",
       free_bytes / 1e9, total_bytes / 1e9);
```

---

### 3.2 Shared Memory (__shared__)

On-chip memory shared by all threads within a thread block. Think of it
as a programmer-managed L1 cache.

```c
__global__ void kernel() {
    // Static shared memory (size known at compile time)
    __shared__ float tile[32][33];  // +1 padding to avoid bank conflicts

    // Dynamic shared memory (size specified at launch)
    extern __shared__ float dynamic_smem[];
}

// Launch with dynamic shared memory:
kernel<<<grid, block, dynamicSharedBytes>>>();
```

**Architecture progression of shared memory:**

| Architecture | Max Shared/SM | Configurable | Notes |
|-------------|---------------|-------------|-------|
| Kepler | 48 KB | Yes (48/16 or 16/48 with L1) | |
| Maxwell | 96 KB | Yes | Dedicated, not shared with L1 |
| Pascal | 64 KB | Yes | |
| Volta | 96 KB | Yes (unified L1/shared, up to 96 KB) | Unified with L1 |
| Turing | 64 KB | Yes | |
| Ampere | 164 KB | Yes (up to 164 KB shared) | Configurable L1/shared |
| Hopper | 228 KB | Yes (up to 228 KB shared) | Largest shared memory |
| Blackwell | 228 KB | Yes | |

**Bank conflicts explained:**

Shared memory is organized into 32 banks (one per warp lane). Each bank
is 4 bytes wide. Successive 4-byte words map to successive banks.

```
Address:   0    4    8   12   16   20   24  ...  124
Bank:     B0   B1   B2   B3   B4   B5   B6  ...  B31

Address: 128  132  136  140  ...
Bank:     B0   B1   B2   B3  ...   (wraps around)
```

**No conflict:** Each thread in a warp accesses a different bank.

```
Thread 0 → Bank 0, Thread 1 → Bank 1, ..., Thread 31 → Bank 31
Result: All 32 accesses served simultaneously in 1 cycle.
```

**N-way conflict:** N threads access the same bank (different addresses).

```
Thread 0 → Bank 0 (addr 0), Thread 16 → Bank 0 (addr 128)
Result: 2-way conflict, serialized into 2 cycles.
```

**Broadcast exception:** If multiple threads read the *same address*
within a bank, it is broadcast (no conflict).

**Avoiding bank conflicts:**

```c
// BAD: 32-way bank conflict (column access of 32-wide array)
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // All threads access bank 0

// GOOD: Add padding to shift bank mapping
__shared__ float data[32][33];     // 33 columns instead of 32
float val = data[threadIdx.x][0];  // Each thread accesses different bank
```

**When to use shared memory:**
- Data reused by multiple threads in a block (tiling)
- Inter-thread communication within a block
- Reduction operations (partial sums)
- Stencil computations (neighbor access)
- Any pattern where global memory would be accessed repeatedly

**When NOT to use:**
- Data accessed only once (shared memory staging adds overhead)
- When you need more than what fits (spills to local/global)
- Simple element-wise operations (no data reuse)

---

### 3.3 Constant Memory (__constant__)

Read-only memory with a dedicated cache optimized for broadcast access
(all threads reading the same address simultaneously).

```c
// Declared at file scope (64 KB limit total)
__constant__ float coefficients[256];

// Write from host:
cudaMemcpyToSymbol(coefficients, host_data, 256 * sizeof(float));

// Read in kernel (fast when all threads read same index):
__global__ void kernel() {
    float c = coefficients[blockIdx.x];  // Broadcast: all threads get same value
}
```

**Cache behavior:**

```
All threads in warp read SAME address:     → 1 cache read, broadcast to all 32
                                              Latency: ~5 cycles (as fast as shared)

All threads in warp read DIFFERENT addresses: → Serialized!
                                                Latency: up to 32 × ~5 cycles = ~160
```

**This means constant memory is ONLY fast when access is uniform within a warp.**

**When to use:**
- Lookup tables read by all threads with the same index
- Kernel configuration parameters (filter weights, physical constants)
- Small read-only arrays (fits in 64 KB)
- Convolution kernels (all threads apply same filter)

**When NOT to use:**
- Per-thread different indices (serializes badly)
- Data larger than 64 KB
- Data that needs to be updated from device code

---

### 3.4 Texture Memory

Texture memory is global memory accessed through a read-only cache
optimized for 2D spatial locality with hardware interpolation.

```c
// Modern texture object API (CUDA 5.0+)
cudaTextureObject_t tex;

// 1. Allocate backing memory (cudaArray for 2D/3D)
cudaArray_t cuArray;
cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
cudaMallocArray(&cuArray, &desc, width, height);

// 2. Copy data to array
cudaMemcpy2DToArray(cuArray, 0, 0, host_data,
                    width * sizeof(float),
                    width * sizeof(float), height,
                    cudaMemcpyHostToDevice);

// 3. Create texture object
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

cudaTextureDesc texDesc = {};
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;  // Hardware bilinear interpolation
texDesc.normalizedCoords = true;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

// 4. Use in kernel
// float val = tex2D<float>(tex, u, v);

// 5. Cleanup
cudaDestroyTextureObject(tex);
cudaFreeArray(cuArray);
```

**When to use:**
- Image processing (2D spatial locality)
- Volume rendering (3D data)
- Lookup tables with interpolation (e.g., transfer functions)
- When access pattern has spatial locality but isn't perfectly coalesced

**When NOT to use:**
- Purely sequential/coalesced access (global memory is faster)
- Write-heavy workloads (texture is read-only; use surface objects for writes)
- When data changes frequently (copying to cudaArray has overhead)

---

### 3.5 Registers

The fastest storage on the GPU. Each thread gets its own registers,
allocated by the compiler.

```
Register budget per SM:
  65,536 × 32-bit registers = 256 KB per SM

If each thread uses 32 registers and block size is 256:
  256 threads × 32 registers = 8,192 registers per block
  65,536 / 8,192 = 8 blocks can fit on one SM

If each thread uses 64 registers:
  256 × 64 = 16,384 registers per block
  65,536 / 16,384 = 4 blocks → lower occupancy
```

**Register pressure** is a major performance concern. Using too many
registers per thread reduces the number of blocks (and warps) that can
be resident on an SM, which reduces the SM's ability to hide memory latency.

**Controlling register usage:**

```c
// Limit registers per thread (trade-off: may spill to local memory)
__global__ __launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks/SM
void kernel() { ... }

// Or via compiler flag:
// nvcc --maxrregcount=32 kernel.cu
```

**Register spilling:** When a thread needs more registers than available,
the compiler spills excess variables to **local memory** (which is actually
global memory, just addressed per-thread). This is slow (~400 cycles).

Check register usage with:
```bash
nvcc --ptxas-options=-v kernel.cu
# Shows: Used 42 registers, 0 bytes smem, 0 bytes cmem
```

---

### 3.6 Local Memory

Local memory is NOT a separate physical memory. It resides in global memory
(HBM/GDDR) and is private to each thread. The compiler uses it when:

- Arrays declared in a kernel are too large for registers
- The compiler cannot determine array indices at compile time
- Register spilling occurs

```c
__global__ void kernel() {
    float local_array[64];     // May be in registers or local memory
    float indexed = local_array[threadIdx.x % 64];  // Index not constant
                                                     // → local memory
}
```

**Performance:** Same latency as global memory (~400 cycles), but cached in
L1 and L2. Accessing local memory is still much slower than registers or
shared memory.

**How to detect local memory usage:**

```bash
nvcc --ptxas-options=-v kernel.cu
# Look for: "Used N registers, M bytes lmem"
# lmem > 0 means local memory is being used
```

**How to reduce local memory usage:**
- Use `__launch_bounds__` to give compiler more information
- Reduce array sizes or use shared memory instead
- Ensure array indices are compile-time constants when possible
- Consider reducing per-thread work and using more threads

---

## 4. Unified / Managed Memory

Unified Memory (UM) provides a single pointer valid on both CPU and GPU.
The CUDA runtime automatically migrates pages between host and device
on demand via page faults.

```c
float *data;
cudaMallocManaged(&data, N * sizeof(float));

// Use on CPU
for (int i = 0; i < N; i++) data[i] = i;

// Use on GPU (automatic migration)
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Use on CPU again (automatic migration back)
printf("%f\n", data[0]);

cudaFree(data);  // Single free
```

**How it works under the hood:**

```
                CPU accesses 'data'
                       │
                       ▼
              ┌────────────────┐
              │ Is page on CPU?│
              └───────┬────────┘
                 Yes  │  No
                 │    │  │
                 │    │  ▼
                 │    │  PAGE FAULT
                 │    │  ├─ Migrate page from GPU → CPU
                 │    │  └─ Resume access
                 ▼    │
             [Access] │
                      │
                GPU accesses 'data'
                      │
                      ▼
              ┌────────────────┐
              │ Is page on GPU?│
              └───────┬────────┘
                 Yes  │  No
                 │    │  │
                 │    │  ▼
                 │    │  PAGE FAULT
                 │    │  ├─ Migrate page from CPU → GPU
                 │    │  └─ Resume access
                 ▼    │
              [Access]│
```

**Performance optimization with hints:**

```c
int device;
cudaGetDevice(&device);

// "This data will mostly be read, not written"
// → Runtime can create read-only copies on multiple devices
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device);

// "Keep this data on the GPU unless absolutely needed on CPU"
// → Reduces unnecessary migration
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device);

// "The GPU will access this data" (even if it's on the CPU)
// → Runtime can set up direct access instead of migrating
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device);

// Explicitly prefetch to GPU (avoids demand-paging latency)
cudaMemPrefetchAsync(data, size, device, stream);

// Prefetch back to CPU
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
```

**Unified Memory pros and cons:**

| Pros | Cons |
|------|------|
| Dramatically simpler code | Page fault overhead on first access |
| Single pointer, no explicit copies | Thrashing if CPU and GPU compete |
| Oversubscription (GPU can use more than VRAM) | Harder to reason about performance |
| Great for prototyping | Prefetching required for peak perf |
| Works with complex data structures (linked lists, trees) | Not available on all GPUs (CC 6.0+) |

**When to use:**
- Prototyping and rapid development
- Complex data structures with pointers (graphs, trees, linked lists)
- When data access patterns are unpredictable
- Multi-GPU programming (simpler than explicit management)
- GPU memory oversubscription (dataset larger than VRAM)

**When NOT to use:**
- Performance-critical hot paths (explicit management is faster)
- Known, predictable access patterns (just use cudaMemcpy)
- Very latency-sensitive applications

---

## 5. Stream-Ordered Allocation

Introduced in CUDA 11.2, stream-ordered allocation ties memory lifetime
to a CUDA stream, avoiding the global synchronization of `cudaMalloc/cudaFree`.

```c
cudaStream_t stream;
cudaStreamCreate(&stream);

float *d_tmp;
cudaMallocAsync(&d_tmp, size, stream);    // Non-blocking, no sync

kernel<<<grid, block, 0, stream>>>(d_tmp);

cudaFreeAsync(d_tmp, stream);             // Non-blocking, reusable
cudaStreamSynchronize(stream);
```

**Why it matters:**

```
Traditional cudaMalloc/cudaFree:
  All streams: ═══╗ cudaMalloc (global sync) ╠══════════════════════
                  ║                          ║
  Stream 0:  ─────╨──────────────────────────╨──────────────────────
  Stream 1:  ─────╨── blocked  ──────────────╨── can resume ────────

cudaMallocAsync/cudaFreeAsync:
  Stream 0:  ─── cudaMallocAsync ─── kernel ─── cudaFreeAsync ────
  Stream 1:  ─── continues working without any interruption ──────
```

**Memory pools:**

Under the hood, `cudaMallocAsync` uses memory pools. The pool retains freed
memory and reuses it for subsequent allocations, avoiding OS round-trips.

```c
// Access the default pool
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, device);

// Configure pool behavior
uint64_t threshold = UINT64_MAX;  // Never release memory back to OS
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

// Create a custom pool with specific properties
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypePinned;
props.handleTypes = cudaMemHandleTypeNone;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = device;
cudaMemPoolCreate(&pool, &props);
```

**When to use:**
- Workloads with many temporary allocations (GNNs, sparse operations)
- Dynamic parallelism (kernels allocating device memory)
- When you need allocation without synchronizing other streams
- Pipeline patterns with varying memory requirements

---

## 6. Zero-Copy Memory

Zero-copy allows the GPU to read/write host memory directly over the
PCIe/NVLink bus without an explicit `cudaMemcpy`.

```c
float *h_data;
cudaHostAlloc(&h_data, size,
              cudaHostAllocMapped);  // Pinned + mapped

// Get device-side pointer
float *d_ptr;
cudaHostGetDevicePointer(&d_ptr, h_data, 0);

// GPU reads/writes host memory directly
kernel<<<grid, block>>>(d_ptr, N);
```

**Performance characteristics:**

```
Access Type          Bandwidth       Latency
──────────────────   ─────────       ───────
Global memory (HBM)  1-8 TB/s       ~400 cycles
Zero-copy (PCIe 4)   ~25 GB/s       ~10,000+ cycles
Zero-copy (PCIe 5)   ~50 GB/s       ~8,000+ cycles
Zero-copy (NVLink 5)  ~1.8 TB/s     Lower (coherent)
```

**When to use:**
- Data accessed only once (no reuse)
- Dataset too large for GPU memory
- Results written once by GPU, read once by CPU
- Integrated GPU systems (e.g., Jetson) where CPU and GPU share memory
- Grace-Blackwell systems with NVLink CPU-GPU (coherent, high bandwidth)

**When NOT to use:**
- Data accessed repeatedly by GPU (copy to device memory instead)
- Latency-sensitive random access patterns
- High-throughput kernels (PCIe bandwidth is the bottleneck)

---

## 7. L2 Cache Persistence (Ampere+)

Starting with Ampere (sm_80), you can reserve a portion of the L2 cache
for specific data, keeping it "warm" across kernel launches.

```c
// Reserve 32 MB of L2 for persistent access
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 32 * 1024 * 1024);

// Create an access policy window
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr = d_hot_data;
attr.accessPolicyWindow.num_bytes = hot_data_size;
attr.accessPolicyWindow.hitRatio = 1.0f;       // Cache all accesses
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// Kernels on this stream will find d_hot_data in L2
kernel<<<grid, block, 0, stream>>>(d_hot_data, ...);

// Reset when done
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();
```

**When to use:**
- Small, frequently-accessed lookup tables
- Data reused across multiple kernel launches
- Working sets smaller than L2 capacity (up to ~40 MB on A100, ~50 MB on H100)

---

## 8. Tensor Memory - TMEM (Blackwell)

Blackwell datacenter GPUs (sm_100) introduce **Tensor Memory (TMEM)**, a
dedicated on-chip memory exclusively for tensor core operands. This is not
directly accessible from CUDA C++ -- it is managed via PTX instructions
(`tcgen05.*`) and typically accessed through CUTLASS 4.x templates.

```
Traditional MMA pipeline:
  Global Memory → Shared Memory → Registers → Tensor Core
                                   ↑ accumulator lives in registers
                                   (consumes register file heavily)

Blackwell TMEM pipeline:
  Global Memory → Shared Memory → TMEM → Tensor Core
                                   ↑ accumulator lives in TMEM
                                   (frees register file for other work)
```

**Key TMEM instructions (PTX level):**

```
tcgen05.alloc           Allocate TMEM region
tcgen05.ld / tcgen05.st Load/store data to/from TMEM
tcgen05.cp              Copy data into TMEM (from shared memory)
tcgen05.commit          Commit pending copy operations
tcgen05.fence           Memory fence for TMEM
tcgen05.wait            Wait for TMEM operations to complete
tcgen05.dealloc         Free TMEM region
tcgen05.mma             Execute matrix multiply-accumulate using TMEM
```

**Practical impact:** TMEM is abstracted away by CUTLASS and CUDA Tile.
You don't write TMEM code directly unless you're authoring PTX-level
kernels. However, understanding that it exists helps explain why Blackwell
tensor core kernels can achieve higher occupancy than equivalent Hopper
kernels -- the accumulator no longer competes for the register file.

---

## 9. Memory Transfer Patterns

### Pattern 1: Basic Synchronous Transfer

```c
cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);   // Blocks CPU
kernel<<<...>>>(d_dst);
cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);   // Blocks CPU
```

Timeline:
```
CPU:  [H2D copy]──wait──[launch]──wait──[D2H copy]──wait──
GPU:  ─────────────────[kernel]────────────────────────────
```

### Pattern 2: Async Transfer with Pinned Memory

```c
cudaMemcpyAsync(d_dst, h_pinned, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_dst);
cudaMemcpyAsync(h_pinned, d_dst, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

### Pattern 3: Double Buffering (Overlapping Transfer + Compute)

```c
// Two buffers, two streams
float *d_buf[2], *h_buf[2];
cudaStream_t stream[2];

for (int i = 0; i < num_chunks; i++) {
    int cur = i % 2;
    int prev = (i + 1) % 2;

    // Transfer chunk i to GPU (stream cur)
    cudaMemcpyAsync(d_buf[cur], h_buf[cur], chunk_size,
                    cudaMemcpyHostToDevice, stream[cur]);

    // Process chunk i-1 on GPU (stream prev, if i > 0)
    if (i > 0)
        kernel<<<grid, block, 0, stream[prev]>>>(d_buf[prev], ...);
}
```

Timeline with double buffering:
```
Stream 0: [H2D chunk 0]          [H2D chunk 2]          [H2D chunk 4]
Stream 1:              [H2D chunk 1]          [H2D chunk 3]
GPU:                [kernel 0] [kernel 1] [kernel 2] [kernel 3]
                    ↑ overlapped with next transfer
```

### Pattern 4: Unified Memory with Prefetching

```c
cudaMallocManaged(&data, size);
// CPU writes
for (...) data[i] = ...;

// Prefetch to GPU before kernel
cudaMemPrefetchAsync(data, size, device, stream);
kernel<<<grid, block, 0, stream>>>(data);

// Prefetch back before CPU reads
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
printf("%f\n", data[0]);
```

---

## 10. Decision Framework

### "Which host memory should I use?"

```
                    ┌─────────────────────┐
                    │ Do you need fast    │
                    │ GPU transfers?      │
                    └─────────┬───────────┘
                         Yes  │  No
                         │    └──→ malloc() (pageable)
                         ▼
                    ┌─────────────────────┐
                    │Will GPU access      │
                    │host memory directly?│
                    └─────────┬───────────┘
                         Yes  │  No
                         │    │
                         │    ▼
                         │  ┌─────────────────────┐
                         │  │ CPU writes only,    │
                         │  │ GPU reads?          │
                         │  └─────────┬───────────┘
                         │       Yes  │  No
                         │       │    └──→ cudaHostAlloc (default)
                         │       └────→ cudaHostAlloc (WriteCombined)
                         │
                         ▼
                    cudaHostAlloc (Mapped) → zero-copy
```

### "Which device memory should I use?"

```
┌───────────────────────────────────────────────────────────────────────┐
│ Data Characteristics             │ Recommended Memory                 │
├───────────────────────────────────────────────────────────────────────┤
│ Large, per-element R/W           │ Global memory (cudaMalloc)         │
│ Reused by threads in a block     │ Shared memory (__shared__)         │
│ Small, read-only, uniform access │ Constant memory (__constant__)     │
│ 2D/3D spatial locality, read-only│ Texture memory                     │
│ Per-thread scalars               │ Registers (automatic)              │
│ Both CPU and GPU need access     │ Unified memory (cudaMallocManaged) │
│ GPU only, temporary, in streams  │ cudaMallocAsync (stream-ordered)   │
│ Accessed once from host          │ Zero-copy (mapped pinned)          │
│ Tensor core operands (Blackwell) │ TMEM (via CUTLASS / PTX)           │
│ Hot working set < L2 size        │ L2 persistence (Ampere+)           │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 11. Common Pitfalls

### Pitfall 1: Forgetting that cudaMemcpy is synchronous

```c
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// CPU is blocked here until transfer completes!
// Use cudaMemcpyAsync + pinned memory + streams for overlap.
```

### Pitfall 2: Using pageable memory with cudaMemcpyAsync

```c
float *h_data = (float*)malloc(size);  // Pageable!
cudaMemcpyAsync(d_data, h_data, size,
                cudaMemcpyHostToDevice, stream);
// This silently falls back to SYNCHRONOUS behavior.
// Must use pinned memory for true async.
```

### Pitfall 3: Allocating too much pinned memory

```c
// Pinning 90% of system RAM → OS instability, OOM killer
cudaHostAlloc(&huge_buffer, 30ULL * 1024 * 1024 * 1024, 0);
// Rule: Pin only the buffers actively used for transfers.
```

### Pitfall 4: Shared memory bank conflicts

```c
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];    // 32-way bank conflict!
// Fix: smem[32][33] (padding), or transpose access pattern.
```

### Pitfall 5: Constant memory with non-uniform access

```c
__constant__ float table[1024];
float val = table[threadIdx.x];  // Each thread reads different index
// Result: Serialized! Up to 32 sequential reads per warp.
// Fix: Use shared memory or global memory instead.
```

### Pitfall 6: Unified Memory thrashing

```c
cudaMallocManaged(&data, size);
for (int iter = 0; iter < 1000; iter++) {
    kernel<<<...>>>(data);            // Migrates to GPU
    cudaDeviceSynchronize();
    printf("%f\n", data[0]);          // Migrates to CPU (page fault!)
}
// Fix: Batch CPU access, or use cudaMemPrefetchAsync.
```

### Pitfall 7: Not checking cudaMalloc return values

```c
float *d_data;
cudaMalloc(&d_data, impossibly_large_size);
kernel<<<...>>>(d_data);  // d_data is NULL → segfault or silent corruption
// Fix: Always use CUDA_CHECK macro (see tutorial 02).
```

### Pitfall 8: Mismatched cudaMalloc/free pairs

```c
// cudaMalloc → cudaFree
// cudaHostAlloc → cudaFreeHost
// cudaMallocManaged → cudaFree
// cudaMallocAsync → cudaFreeAsync
// malloc → free
// DO NOT MIX THEM.
```

---

## 12. API Quick Reference

### Host Memory

```c
// Pageable (standard C)
void *ptr = malloc(size);
free(ptr);

// Pinned
cudaHostAlloc(&ptr, size, flags);          // Allocate pinned
cudaFreeHost(ptr);                         // Free pinned

// Register existing allocation as pinned
cudaHostRegister(ptr, size, flags);        // Pin existing buffer
cudaHostUnregister(ptr);                   // Unpin

// Get device pointer for mapped memory
cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
```

### Device Memory

```c
// Global
cudaMalloc(&d_ptr, size);                  // Allocate
cudaFree(d_ptr);                           // Free
cudaMalloc3D(&pitchedPtr, extent);         // 3D allocation (pitched)
cudaMallocPitch(&d_ptr, &pitch, w, h);    // 2D allocation (pitched)

// Stream-ordered
cudaMallocAsync(&d_ptr, size, stream);     // Async allocate
cudaFreeAsync(d_ptr, stream);              // Async free

// Unified
cudaMallocManaged(&ptr, size);             // Both CPU+GPU
cudaFree(ptr);                             // Free
```

### Transfers

```c
cudaMemcpy(dst, src, size, direction);                  // Synchronous
cudaMemcpyAsync(dst, src, size, direction, stream);     // Async (needs pinned)
cudaMemcpy2D(dst, dpitch, src, spitch, w, h, dir);     // 2D (pitched)
cudaMemcpyToSymbol(symbol, src, size);                  // Host → constant
cudaMemcpyFromSymbol(dst, symbol, size);                // Constant → host
cudaMemset(d_ptr, value, size);                         // Set bytes
cudaMemsetAsync(d_ptr, value, size, stream);            // Async set
cudaMemPrefetchAsync(ptr, size, device, stream);        // Unified mem hint
cudaMemAdvise(ptr, size, advice, device);               // Unified mem advice
```

### Memory Information

```c
cudaMemGetInfo(&free, &total);             // Available GPU memory
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
  prop.totalGlobalMem                      // Total global memory
  prop.sharedMemPerBlock                   // Max shared memory per block
  prop.sharedMemPerMultiprocessor          // Max shared memory per SM
  prop.regsPerBlock                        // Max registers per block
  prop.regsPerMultiprocessor               // Max registers per SM
  prop.totalConstMem                       // Total constant memory (64 KB)
  prop.l2CacheSize                         // L2 cache size
  prop.memoryBusWidth                      // Memory bus width (bits)
  prop.memoryClockRate                     // Memory clock (kHz)
```

---

## Further Reading

- [CUDA C++ Programming Guide: Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [CUDA C++ Best Practices Guide: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Nsight Compute: Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/)
- Example files: [`examples/03_memory_model.cu`](examples/03_memory_model.cu), [`examples/06_shared_memory.cu`](examples/06_shared_memory.cu), [`examples/17_advanced_memory.cu`](examples/17_advanced_memory.cu)

---

*This guide is part of the CUDA Know-Hows tutorial collection.*
*See README.md for the complete tutorial index.*
