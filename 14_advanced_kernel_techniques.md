# 14 — Advanced Kernel Techniques

> Part of **[CUDA Know-Hows](README.md)**. Prev: [13 — Streams & concurrency](13_streams_and_concurrency.md).
> Next: [15 — Advanced memory techniques](15_advanced_memory_techniques.md).
> Runnable code: [`examples/08_advanced_topics.cu`](examples/08_advanced_topics.cu).
>
> Goal: the toolkit that separates competent from expert kernels — **warp-level
> primitives** (shuffle/vote/reduce), **cooperative groups**, **dynamic
> parallelism**, and useful intrinsics. These let warps and blocks cooperate
> without shared memory or `__syncthreads()`, and let kernels launch kernels.

---

## 1. Warp-level primitives: cooperate without shared memory

Threads in a warp execute together, so they can exchange registers directly —
faster than round-tripping through shared memory, and needing no barrier. All
take a **mask** of participating lanes (use `__activemask()` or `0xffffffff` when
the whole warp is active) and are the `*_sync` forms (Volta+ requires the mask).

```
   __shfl_sync(mask, val, srcLane)      read val from a specific lane
   __shfl_down_sync(mask, val, delta)   read val from lane (self + delta)
   __shfl_up_sync(mask, val, delta)     read val from lane (self - delta)
   __shfl_xor_sync(mask, val, laneMask) read val from lane (self XOR laneMask) -> butterfly
   __ballot_sync(mask, pred)            bitmask of which lanes have pred != 0
   __any_sync / __all_sync(mask, pred)  did ANY / ALL lanes satisfy pred?
   __reduce_add_sync(mask, val)         hardware warp reduction (sm_80+)
   __match_any_sync(mask, val)          lanes with the same value (sm_70+)
```

### Warp reduction with shuffle (the idiom to memorize)

```cpp
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);  // butterfly-style tree
    return val;                          // lane 0 holds the warp's total
}
```

```
   SHUFFLE-DOWN REDUCTION (8 of 32 lanes shown):

   lanes:   v0 v1 v2 v3 v4 v5 v6 v7
   off=4:   v0+=v4  v1+=v5  v2+=v6  v3+=v7          (add lane+4)
   off=2:   v0+=v2  v1+=v3                          (add lane+2)
   off=1:   v0+=v1                                  (add lane+1)
   result:  v0 = sum of the warp. NO shared memory, NO __syncthreads().
```

```
   BLOCK reduction = warp-reduce each warp -> write 1 value/warp to shared ->
   warp-reduce those. Two shuffle reductions + a tiny shared array. This is how
   fast reductions are written (and what CUB does under the hood).
```

---

## 2. Cooperative Groups: portable, explicit cooperation

Cooperative Groups (`<cooperative_groups.h>`) is a modern API that makes the
*group* of cooperating threads explicit and composable — at warp, tile, block,
or even grid scope — instead of relying on implicit warp behavior.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k() {
    cg::thread_block block = cg::this_thread_block();       // the whole block
    block.sync();                                            // == __syncthreads()

    // Partition into fixed-size tiles (e.g. 32 = a warp, or smaller sub-warps):
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int laneSum = cg::reduce(warp, myVal, cg::plus<int>());  // clean warp reduce
    warp.shfl_down(myVal, 1);                                // typed shuffle
}
```

```
   COOPERATIVE GROUPS give you:
     - explicit groups: this_thread_block(), tiled_partition<N>(), coalesced_threads()
     - group.sync(), group.thread_rank(), group.size()
     - cg::reduce / cg::inclusive_scan over a group (readable, correct)
     - GRID groups: cg::this_grid().sync() for GRID-WIDE sync in ONE launch
       (requires cudaLaunchCooperativeKernel + a co-resident grid)
   Prefer these over raw shuffles for new code: clearer intent, sub-warp safe,
   and portable across the divergence rules of Volta+.
```

### Grid-wide synchronization (single-kernel iterative algorithms)

Normally blocks can't sync with each other within a launch. A **cooperative
launch** lets the whole grid sync, enabling iterative algorithms (e.g. some graph
/ solver kernels) without relaunching:

```cpp
cg::grid_group grid = cg::this_grid();
// ... phase 1 ...
grid.sync();          // ALL blocks reach here before any continues
// ... phase 2 ...
// launched via cudaLaunchCooperativeKernel; grid must fit co-resident on the GPU
```

---

## 3. Dynamic Parallelism: kernels launching kernels

A kernel can launch child kernels directly from the device — useful for
*recursive* or *data-dependent* work (adaptive mesh refinement, tree/graph
traversal, quicksort) where the amount of work isn't known until the GPU is
running.

```cpp
__global__ void child(int* data, int n) { /* ... */ }

__global__ void parent(int* data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // launch a child grid from the DEVICE
        child<<<(n+255)/256, 256>>>(data, n);
        // CUDA Dynamic Parallelism 2 (CUDA 12+) changed sync semantics:
        // device-side cudaDeviceSynchronize() is removed; use tail launches or
        // stream/event ordering to sequence child work.
    }
}
```

```bash
# Requires relocatable device code + the device runtime library:
nvcc -arch=sm_70 -rdc=true prog.cu -o prog -lcudadevrt
```

```
   DYNAMIC PARALLELISM:
     parent kernel ──launches──▶ child kernel(s) ──▶ grandchildren ...
   + expresses recursion / data-dependent parallelism naturally on-device
   - launch overhead per child; can be slower than a flat kernel — measure!
   Best when work is genuinely irregular/recursive and unknown ahead of time.
```

---

## 4. Useful intrinsics & math

```
   MATH INTRINSICS (fast, lower precision — like CPU -ffast-math per op):
     __fmul_rn, __fadd_rn, __fdividef, __expf, __logf, __sinf, __powf
     rsqrtf (fast reciprocal sqrt), __saturatef (clamp to [0,1])
     enable globally with --use_fast_math, or call the __ intrinsics explicitly.

   FUSED MULTIPLY-ADD: fmaf(a,b,c) = a*b+c with ONE rounding (more accurate + fast).

   BIT / INTEGER: __popc (popcount), __clz (count leading zeros), __ffs,
                  __brev (bit reverse), __byte_perm, __funnelshift.

   INTEGER DIVISION is slow on GPUs — precompute reciprocals or use bit ops for
   power-of-two divisors. atomicCAS-based tricks for custom atomics (Ch. 12).
```

---

## 5. Loop unrolling & launch bounds

```cpp
#pragma unroll                    // fully unroll (compile-time trip count)
for (int k = 0; k < TILE; ++k) sum += a[k]*b[k];

#pragma unroll 4                  // unroll by 4
for (int i = 0; i < n; ++i) ...

// Tell the compiler your launch config so it allocates registers for occupancy:
__global__ void __launch_bounds__(256, 4)   // maxThreadsPerBlock, minBlocksPerSM
kernel(...) { ... }
```

```
   __launch_bounds__(T, B) caps registers so >= B blocks of T threads fit per SM,
   trading per-thread registers for occupancy. Combine with -Xptxas -v to verify
   register usage and with Nsight to confirm it actually helped (Ch. 08, 18).
```

---

## 6. Persistent kernels (advanced pattern)

Instead of launching a kernel per batch, a **persistent kernel** keeps a fixed
grid resident and pulls work from a queue in a loop — amortizing launch overhead
and enabling producer/consumer patterns on the GPU. Powerful but tricky
(requires careful synchronization and often cooperative groups); measure vs plain
relaunching or CUDA Graphs (Ch. 16) before adopting.

---

## 7. Key takeaways

- **Warp primitives** (`__shfl_*_sync`, `__ballot_sync`, `__reduce_*_sync`) let a
  warp cooperate via registers — faster than shared memory, no barrier. The
  shuffle reduction is a must-know idiom.
- **Cooperative Groups** make cooperation explicit, sub-warp-safe, and portable
  (`tiled_partition`, `cg::reduce`, group `.sync()`), including **grid-wide sync**
  via cooperative launch. Prefer them for new code.
- **Dynamic parallelism** lets kernels launch kernels for recursive/data-dependent
  work (`-rdc=true -lcudadevrt`); measure — it has overhead.
- Use **fast math intrinsics**, `fmaf`, and bit intrinsics judiciously; avoid slow
  integer division.
- Guide the compiler with **`#pragma unroll`** and **`__launch_bounds__`**; verify
  with `-Xptxas -v` and Nsight.

**Next:** [15 — Advanced memory techniques →](15_advanced_memory_techniques.md)
