# 03 — Thread Hierarchy

> Part of **[CUDA Know-Hows](README.md)**. Prev: [02 — Your first kernel](02_first_kernel.md).
> Next: [04 — Thread indexing patterns](04_thread_indexing_patterns.md).
> Runnable code: [`examples/04_thread_organization.cu`](examples/04_thread_organization.cu).
>
> Goal: master how CUDA organizes threads — thread → warp → block → grid (→
> cluster on Hopper+) — how `dim3` builds 1D/2D/3D launches, how to compute a
> global index in any dimension, and the grid-stride loop that decouples data
> size from launch size.

---

## 1. The hierarchy

CUDA threads are organized into a strict nesting. You choose the grid and block
shapes at launch; the hardware groups threads into **warps** of 32 for execution.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ GRID  (one kernel launch)                                               │
   │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
   │   │ BLOCK (0)     │  │ BLOCK (1)     │  │ BLOCK (2)     │  ...          │
   │   │  ┌─────────┐  │  │  ┌─────────┐  │  │               │               │
   │   │  │ WARP 0  │  │  │  │ WARP 0  │  │  │   a block =   │               │
   │   │  │ 32 thr  │  │  │  │ 32 thr  │  │  │   up to 1024  │               │
   │   │  ├─────────┤  │  │  ├─────────┤  │  │   threads =   │               │
   │   │  │ WARP 1  │  │  │  │ WARP 1  │  │  │   up to 32    │               │
   │   │  │ 32 thr  │  │  │  │ ...     │  │  │   warps       │               │
   │   │  └─────────┘  │  │  └─────────┘  │  │               │               │
   │   └───────────────┘  └───────────────┘  └───────────────┘               │
   └─────────────────────────────────────────────────────────────────────────┘

   THREAD : runs the kernel; has private registers.
   WARP   : 32 threads that execute in lockstep (SIMT). The scheduling unit.
   BLOCK  : up to 1024 threads; share SHARED MEMORY and can __syncthreads().
            A block runs entirely on ONE SM (streaming multiprocessor).
   GRID   : all blocks of a launch. Blocks are INDEPENDENT (no ordering, no
            block-to-block sync within a launch — that's by design, for scaling).
   CLUSTER (Hopper sm_90+, optional): a group of blocks that CAN cooperate via
            distributed shared memory. (Chapter 21.)
```

```
   WHY THIS STRUCTURE? Independence of blocks is what lets the SAME binary scale
   across a tiny GPU (few SMs, blocks run in waves) and a huge one (many SMs,
   blocks run at once) with no code change:

     small GPU (2 SMs):  [B0][B1] then [B2][B3] then [B4][B5]   (3 waves)
     big GPU   (6 SMs):  [B0][B1][B2][B3][B4][B5]               (1 wave)
```

---

## 2. Built-in index variables

Inside a kernel, four built-in `dim3`-typed variables (each with `.x/.y/.z`) tell
a thread who it is and how big its world is:

```
   ┌──────────────┬────────────────────────────────────────────────────────────┐
   │ threadIdx    │ this thread's index WITHIN its block   (0 .. blockDim-1)   │
   │ blockDim     │ block dimensions = threads per block   (set at launch)     │
   │ blockIdx     │ this block's index WITHIN the grid      (0 .. gridDim-1)   │
   │ gridDim      │ grid dimensions = blocks per grid       (set at launch)    │
   └──────────────┴────────────────────────────────────────────────────────────┘
   Also useful:  warpSize  (== 32 on all current NVIDIA GPUs).
```

The universal 1D global index:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
//      └── blocks before me ──┘   └─ my slot ─┘
```

```
   Example: blockDim.x = 4, threadIdx across blocks:

   block 0        block 1        block 2
   t0 t1 t2 t3    t0 t1 t2 t3    t0 t1 t2 t3      (threadIdx.x resets per block)
   ▼  ▼  ▼  ▼     ▼  ▼  ▼  ▼     ▼  ▼  ▼  ▼
   0  1  2  3     4  5  6  7     8  9 10 11       (global index i, continuous)
                  └ blockIdx.x=1 * blockDim.x=4 = 4, + threadIdx.x
```

---

## 3. `dim3` and multi-dimensional launches

`dim3` is a simple 3-component struct. Unspecified components default to 1, so
`dim3 b(256)` means `(256,1,1)`. Use 2D/3D shapes when your *data* is 2D/3D
(images, matrices, volumes) — it makes indexing natural, not faster per se.

```cpp
// 1D launch: N elements
dim3 block(256);
dim3 grid((N + 255) / 256);
kernel1d<<<grid, block>>>(...);

// 2D launch: WxH image, 16x16 threads per block
dim3 block(16, 16);                                   // 256 threads
dim3 grid((W + 15) / 16, (H + 15) / 16);
kernel2d<<<grid, block>>>(...);

// 3D launch: WxHxD volume, 8x8x8 threads per block
dim3 block(8, 8, 8);                                  // 512 threads
dim3 grid((W+7)/8, (H+7)/8, (D+7)/8);
kernel3d<<<grid, block>>>(...);
```

Computing global coordinates in each dimension:

```cpp
// 2D:
int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
if (x < W && y < H) {
    int idx = y * W + x;                          // row-major flatten
    out[idx] = ...;
}

// 3D:
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x < W && y < H && z < D)
    out[(z*H + y)*W + x] = ...;                   // z-major flatten
```

```
   2D GRID OF 2D BLOCKS (W=8, H=6, block 4x3):

        blockIdx.x=0        blockIdx.x=1
      ┌───────────────┐   ┌───────────────┐
   by │ (0,0) threads │   │ (0,0) threads │   each cell below is one thread;
   =1 │ x:0-3 y:3-5   │   │ x:4-7 y:3-5   │   global (x,y) = blockIdx*blockDim
      ├───────────────┤   ├───────────────┤            + threadIdx
   by │ (0,0) threads │   │ (0,0) threads │
   =0 │ x:0-3 y:0-2   │   │ x:4-7 y:0-2   │
      └───────────────┘   └───────────────┘
   Chapter 04 covers indexing patterns (and the coalescing implications) in depth.
```

---

## 4. Bounds checking and choosing block size

You almost never have a data size that's an exact multiple of the block size, so
you launch *slightly too many* threads and guard with an `if`. Launching extra
*threads* is cheap; launching whole idle *blocks* is wasteful, so size the grid
with a ceiling divide.

```cpp
int threads = 256;
int blocks  = cuda::ceil_div(n, threads);     // == (n + threads - 1) / threads
kernel<<<blocks, threads>>>(data, n);         // kernel does `if (i < n)`
```

```
   BLOCK SIZE RULES OF THUMB (full treatment in Ch. 09 Work Allocation):
     - must be a multiple of 32 (warp size) — else you waste lanes in a warp
     - 128 or 256 is a great default; 256 is the most common starting point
     - max 1024 threads per block (hardware limit)
     - bigger isn't always better: more threads/block can mean fewer resident
       blocks per SM (register/shared-mem limits) -> lower flexibility
     - the SM needs MANY warps resident to hide memory latency (Ch. 08)
```

---

## 5. Grid-stride loops (decouple data size from launch size)

Instead of "one thread per element," a **grid-stride loop** lets each thread
process *many* elements, striding by the total number of threads. This one
pattern makes kernels robust to any `n`, tunable in launch size, and friendly to
occupancy tuning and CUDA Graphs.

```cpp
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int stride = blockDim.x * gridDim.x;                 // total threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;  // my start
         i < n;
         i += stride)                                     // jump by all threads
    {
        y[i] = a * x[i] + y[i];
    }
}
```

```
   GRID-STRIDE LOOP with 8 total threads over 20 elements:

   elements:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
   thread 0:  ●              ●              ●                             (0,8,16)
   thread 1:     ●              ●              ●                          (1,9,17)
   thread 2:        ●              ●              ●                       (2,10,18)
   ...        each thread strides by 8, covering all 20 elements in 3 passes.

   BENEFITS:
     - handles ANY n with a FIXED grid size (launch e.g. #SMs * blocks/SM)
     - reuses thread setup / registers across elements
     - contiguous within a warp each step -> COALESCED (Ch. 05)
     - the recommended default pattern for elementwise kernels
```

---

## 6. Warps: the unit that actually executes

Though you think in threads and blocks, the hardware executes **warps** of 32
threads in lockstep. This is the single most important execution fact for
performance, expanded in Chapter 08.

```
   A block of 256 threads = 8 warps (256 / 32).
   Thread t belongs to warp (t / 32), lane (t % 32).

   block threads:  0 .. 31 | 32 .. 63 | 64 .. 95 | ... | 224 .. 255
                    warp 0     warp 1     warp 2           warp 7

   CONSEQUENCES (previews of Ch. 05 & 08):
     - COALESCING: the 32 threads of a warp issue memory together; if they touch
       consecutive addresses, the hardware serves them in few wide transactions.
     - DIVERGENCE: if lanes in a warp take different `if` branches, the warp runs
       both paths with inactive lanes masked -> lost throughput.
     - always make block size a MULTIPLE OF 32 so no warp is partially filled.
```

---

## 7. Limits worth memorizing (query them, don't assume)

```
   Typical hardware limits (query with cudaGetDeviceProperties):
     max threads per block        : 1024
     max block dims               : (1024, 1024, 64)
     max grid dims                : (2^31 - 1, 65535, 65535)
     warp size                    : 32
     max resident blocks per SM   : 16-32 (arch dependent)
     max resident warps per SM    : 48-64
     registers per SM             : 65536
     shared memory per block      : up to 48 KB default (more via opt-in, Ch. 07)
```

```cpp
cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
printf("SMs: %d, maxThreads/block: %d, warpSize: %d, sharedMem/block: %zu\n",
       p.multiProcessorCount, p.maxThreadsPerBlock, p.warpSize, p.sharedMemPerBlock);
```

---

## 8. Key takeaways

- The hierarchy is **thread → warp (32) → block (≤1024) → grid** (→ cluster on
  Hopper+). Blocks are **independent**, which is what lets code scale across GPUs.
- Find your data with `i = blockIdx.x*blockDim.x + threadIdx.x`; extend the same
  pattern per dimension for 2D/3D using **`dim3`**.
- Size the grid with **`cuda::ceil_div`** and always **bounds-check** (`if (i<n)`);
  make block size a **multiple of 32**, default 128/256.
- Prefer **grid-stride loops** to decouple data size from launch size and keep
  accesses coalesced.
- Everything ultimately runs as **warps** — remember coalescing and divergence
  (Chapters 05, 08).

**Next:** [04 — Thread indexing patterns →](04_thread_indexing_patterns.md)
