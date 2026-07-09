# 21 — Modern CUDA (Tensor Cores, Tile, FP8/FP4, Hopper/Blackwell)

> Part of **[CUDA Know-Hows](README.md)**. Prev: [20 — Libraries & ecosystem](20_libraries_and_ecosystem.md).
> Next: [22 — Applications](22_applications.md).
> Runnable code: [`examples/22_modern_cuda.cu`](examples/22_modern_cuda.cu).
>
> Goal: the state of CUDA in 2025–2026 — Tensor Cores and the precisions that
> power AI (FP16/BF16/TF32/FP8/FP4), thread-block clusters, the TMA, and the
> shift from thread-centric to **tile-centric** programming (CUDA Tile C++). This
> is where the field is heading; the fundamentals from Ch. 00–17 still underpin
> all of it.

---

## 1. Tensor Cores: matrix math as a hardware primitive

Since Volta, SMs contain **Tensor Cores** that compute a small **matrix
multiply-accumulate (MMA)** `D = A·B + C` in one operation, at many times the
FP32 throughput. They are why GPUs dominate deep learning.

```
   CUDA core:   one scalar FMA per lane per cycle.
   Tensor Core: a whole small MATRIX MMA per operation:

        A (m x k)     B (k x n)     C (m x n)
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │         │ · │         │ + │         │  = D   (all in ONE hardware op,
      └─────────┘   └─────────┘   └─────────┘         accumulate in higher precision)

   Throughput: often 8-16x FP32 for the same silicon -> the AI performance story.
```

The precisions Tensor Cores accelerate (lower precision = more throughput +
less memory, at reduced numeric range/precision):

```
   ┌────────┬─────────────────────────────────────────────────────────────────────┐
   │ FP16   │ half precision; classic mixed-precision training (Volta+)           │
   │ BF16   │ bfloat16: FP32's exponent range, fewer mantissa bits — training     │
   │ TF32   │ 19-bit internal format; drop-in ~FP32 accuracy at Tensor-Core speed │
   │ FP8    │ e4m3 / e5m2 (Hopper/Ada+): LLM training & inference                 │
   │ FP6/FP4│ (Blackwell): extreme inference throughput, hardware micro-scaling   │
   │ INT8/4 │ quantized inference (Turing+)                                       │
   └────────┴─────────────────────────────────────────────────────────────────────┘
   Lower precision -> more FLOPs + less memory/bandwidth, but you must manage
   numeric range (loss scaling, per-block scaling factors on Blackwell).
```

---

## 2. How you actually use Tensor Cores

You almost never write raw MMA instructions. In order of preference:

```
   1. LIBRARIES (Ch. 20): cuBLASLt, cuDNN, CUTLASS, TensorRT already use Tensor
      Cores optimally. This covers ~all standard GEMM/conv/attention.
   2. CUTLASS / CuTe: templated C++ to build CUSTOM fused Tensor Core kernels at
      library-class speed (when you need fusion the libraries don't offer).
   3. CUDA Tile C++ (below): the new high-level tile programming model.
   4. WMMA API (nvcuda::wmma): the older warp-level fragment API — educational,
      but CUTLASS/Tile supersede it for real work.
```

The classic warp-level WMMA sketch (to understand the model):

```cpp
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16,16,16, half, row_major> a;
fragment<matrix_b, 16,16,16, half, col_major> b;
fragment<accumulator, 16,16,16, float> c;
fill_fragment(c, 0.0f);
load_matrix_sync(a, aPtr, 16);
load_matrix_sync(b, bPtr, 16);
mma_sync(c, a, b, c);                 // Tensor Core MMA on a 16x16x16 tile
store_matrix_sync(cPtr, c, 16, mem_row_major);
```

```
   A WARP cooperatively owns the matrix "fragments"; mma_sync issues the Tensor
   Core op. Hopper's WGMMA extends this to a WARP GROUP (4 warps / 128 threads);
   Blackwell's tcgen05 moves to single-thread-issued MMA with Tensor Memory (TMEM).
```

---

## 3. Thread-block clusters (Hopper sm_90+)

Clusters add a **new level to the hierarchy** (Ch. 03): a group of blocks
guaranteed to be co-resident on neighboring SMs, able to access each other's
shared memory (**distributed shared memory, DSMEM**) and synchronize.

```
   grid ─▶ CLUSTER ─▶ block ─▶ warp ─▶ thread     (cluster = new, optional level)

   ┌──────────── cluster ─────────────┐
   │  ┌ block 0 ┐   ┌ block 1 ┐       │  blocks in a cluster can:
   │  │ smem ◀──┼───┼──▶ smem │       │   - read each other's shared memory (DSMEM)
   │  └─────────┘   └─────────┘       │   - cluster.sync() across blocks
   └──────────────────────────────────┘   - share a larger effective on-chip tile
```

```cpp
// Launch with a cluster dimension (via __cluster_dims__ or cudaLaunchKernelEx):
__global__ void __cluster_dims__(2,1,1) kernel(...) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();                              // sync across blocks in the cluster
    float* remote = cluster.map_shared_rank(smem, otherBlockRank);  // DSMEM access
}
```

Clusters let a "super-block" stage a bigger tile across SMs — used by fast GEMM
and attention kernels alongside TMA.

---

## 4. The Tensor Memory Accelerator (recap) & TMEM

- **TMA** (Ch. 15): hardware engine for async multi-dimensional global↔shared bulk
  copies, issued by one thread via a descriptor.
- **Tensor Memory (TMEM)** (Blackwell): dedicated on-chip memory holding Tensor
  Core operands/accumulators, reducing register/shared-memory pressure during
  large MMAs. Managed by the new `tcgen05` MMA path.

```
   Blackwell matmul dataflow (conceptual):
     global ──TMA──▶ shared ──▶ TMEM ──tcgen05 MMA──▶ TMEM(accum) ──▶ shared ──▶ global
   Copies, staging, and accumulation are increasingly HARDWARE-managed, freeing
   the CUDA cores. You access this through CUTLASS / CUDA Tile, not by hand.
```

---

## 5. CUDA Tile C++ — thread-centric → tile-centric

CUDA Tile (CUDA 13.x, C++20, sm_80+) is a higher-level model where you program in
terms of **tiles of data** and tile operations; the compiler maps them to
threads, shared memory, TMA, and Tensor Cores automatically — across
architectures.

```
   THREAD-CENTRIC (Ch. 02-17): you manage every thread, index, __shared__ tile,
     __syncthreads, and (for MMA) fragments. Maximum control, maximum complexity.

   TILE-CENTRIC (CUDA Tile): you declare tiles and express tile-level math;
     the compiler chooses the thread mapping, staging, and Tensor Core usage.

     load tile A, tile B  ─▶  tile C = matmul(A, B)  ─▶  store tile C
     (the compiler emits the coalesced loads, TMA, shared staging, MMA, epilogue)
```

```
   WHY IT MATTERS:
     + much less boilerplate for high-performance tiled/Tensor-Core kernels
     + portable across Ampere/Ada/Hopper/Blackwell (compiler picks the best path)
     + you still NEED the fundamentals (this whole guide) to reason about perf and
       to know when the abstraction leaks.
   Build with: nvcc -std=c++20 --enable-tile -arch=sm_90a (toolkit 13.x+).
```

---

## 6. Architecture cheat: Hopper vs Blackwell (what's new for programmers)

```
   HOPPER (sm_90, H100/H200):
     - 4th-gen Tensor Cores + FP8
     - Thread-block CLUSTERS + distributed shared memory
     - TMA (Tensor Memory Accelerator)
     - WGMMA (warp-group MMA, 128 threads)
     - DPX instructions (dynamic programming)

   BLACKWELL (sm_100 datacenter / sm_120 consumer, B200/GB200/RTX 50xx):
     - 5th-gen Tensor Cores: native FP4/FP6 with hardware micro-scaling
     - Tensor Memory (TMEM) + tcgen05 single-thread-issued MMA
     - 2nd-gen Transformer Engine (FP4 inference/training)
     - NVLink 5 (1.8 TB/s), dual-die design, 192GB HBM3e
     - CTA-pair execution (two blocks share operands)
   Consumer Blackwell (sm_120) supports block-scaled MMA but not the full
   datacenter tcgen05/TMEM feature set — target sm_XXa for arch-specific features.
```

---

## 7. Practical guidance for modern kernels

```
   - Use mixed precision via LIBRARIES first (cuBLASLt/cuDNN/CUTLASS/TE). Reach for
     raw Tensor Core APIs only for custom fused kernels.
   - Pick the LOWEST precision that meets accuracy needs (BF16/TF32 for training,
     FP8/FP4 for inference) — it's free memory + speed if the numerics hold.
   - On Hopper/Blackwell, performance comes from TMA + clusters + async pipelines
     (Ch. 15) feeding Tensor Cores — again, mostly via CUTLASS/Tile.
   - Try CUDA Tile C++ for new high-performance kernels to cut boilerplate; keep
     the thread-centric mental model for debugging and reasoning.
   - Always profile (Ch. 18): Tensor Core utilization and memory throughput are
     the numbers that matter on modern GPUs.
```

Run the modern-features example (guards features by compute capability):

```bash
cd examples && make 22_modern_cuda && ./22_modern_cuda
```

---

## 8. Key takeaways

- **Tensor Cores** do a matrix MMA in hardware at many× FP32 throughput; they and
  low precisions (**FP16/BF16/TF32/FP8/FP4**) are the AI performance story.
- Use them via **libraries → CUTLASS/CuTe → CUDA Tile → WMMA/WGMMA/tcgen05** (in
  that order of preference); pick the lowest precision that keeps accuracy.
- **Thread-block clusters** (Hopper+) add a hierarchy level with **distributed
  shared memory** and cluster sync; **TMA** and **TMEM** (Blackwell) hardware-manage
  data movement and MMA operands.
- **CUDA Tile C++** shifts programming from threads to **tiles**, letting the
  compiler map to Tensor Cores/TMA portably — but the fundamentals still govern
  performance.
- Know the **Hopper vs Blackwell** feature deltas and target `sm_XXa` for
  arch-specific features.

**Next:** [22 — Applications →](22_applications.md)
