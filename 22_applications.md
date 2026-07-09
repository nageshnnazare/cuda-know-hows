# 22 — Applications

> Part of **[CUDA Know-Hows](README.md)**. Prev: [21 — Modern CUDA](21_modern_cuda.md).
> Next: [99 — Cheatsheet](99_cheatsheet.md).
>
> Goal: see the whole guide applied. Each domain below is a complete, runnable
> example in [`examples/`](examples/), annotated with which chapters' techniques it
> uses. Read the chapter, then read the code, then modify it — that's how the
> concepts become instinct.

---

## How to use this chapter

Each application is a `.cu` file with heavy in-code teaching comments. Build and
run any of them from `examples/` (set `ARCH=sm_XX` for your GPU):

```bash
cd examples
make 12_image_processing && ./12_image_processing
```

The value is in the **mapping**: every one of these is just the fundamentals
(coalescing, tiling, reductions, atomics, streams) applied to a real problem. As
you read each, ask "which golden rule (README) is this exploiting?"

---

## 1. Image processing — [`examples/12_image_processing.cu`](examples/12_image_processing.cu)

Convolution (blur/sharpen/edge detection), grayscale, and filters over 2D images.

```
   CONCEPTS APPLIED:
     - 2D thread indexing (Ch. 03/04): pixel (x,y) per thread
     - shared-memory TILING WITH HALOS (Ch. 07): load a tile + border once, each
       output pixel reads neighbors from fast shared memory
     - constant memory (Ch. 05): the filter kernel is read uniformly -> broadcast
     - coalescing (Ch. 05): row-major access so warps read contiguous pixels
     - texture memory option (Ch. 05): hardware boundary handling & 2D cache

   3x3 CONVOLUTION per output pixel:
     out[y][x] = Σ  filter[i][j] * in[y+i-1][x+j-1]
                i,j
     each pixel reuses a 3x3 neighborhood -> tiling pays off hugely.
```

---

## 2. Sorting — [`examples/13_sorting_algorithms.cu`](examples/13_sorting_algorithms.cu)

Bitonic sort, radix sort, merge — parallel sorting from scratch.

```
   CONCEPTS APPLIED:
     - warp/block cooperation and __syncthreads (Ch. 07/14)
     - scan/prefix-sum as a building block for radix sort (Ch. 07)
     - coalesced global access and shared-memory staging (Ch. 05/07)
     - divergence awareness in compare-exchange networks (Ch. 08)

   BITONIC SORT network (data-independent -> GPU-friendly, O(n log^2 n)):
     compare-exchange elements at fixed strides across log^2(n) stages;
     all threads follow the SAME pattern -> minimal divergence.

   IN PRODUCTION: use cub::DeviceRadixSort / thrust::sort (Ch. 20) — they're
   faster. Hand-roll here to LEARN the parallel patterns.
```

---

## 3. Scientific computing — [`examples/14_scientific_computing.cu`](examples/14_scientific_computing.cu)

PDE stencils, N-body simulation, Monte Carlo integration.

```
   CONCEPTS APPLIED:
     - stencils via shared-memory tiling with halos (Ch. 07)
     - N-body: tiling of body positions in shared memory (each block loads a tile
       of bodies, all threads compute forces against it) — a reduction-like reuse
     - cuRAND for Monte Carlo random sampling (Ch. 20)
     - FMA and math intrinsics; watch FP64 throughput on consumer GPUs (Ch. 08)
     - grid-stride loops for arbitrary problem sizes (Ch. 03)

   N-BODY TILING: O(N^2) interactions, but each loaded position is reused by all
   threads in the block -> shared memory turns it compute-bound.
```

---

## 4. Graph algorithms — [`examples/16_graph_algorithms.cu`](examples/16_graph_algorithms.cu)

BFS, connected components, PageRank — irregular, data-dependent parallelism.

```
   CONCEPTS APPLIED:
     - atomics for frontier updates / visited flags (Ch. 12)
     - the challenge of LOAD IMBALANCE & DIVERGENCE (Ch. 08): vertices have wildly
       different degrees -> naive one-thread-per-vertex diverges badly
     - CSR sparse format + coalescing considerations (Ch. 05)
     - sometimes cooperative groups / grid sync for level-synchronous BFS (Ch. 14)

   GRAPHS are the HARD case: irregular memory access and divergence fight the GPU's
   strengths. Techniques: frontier-based BFS, work-queue load balancing, and
   libraries (nvGRAPH successor / cuGraph) for production.
```

---

## 5. Machine-learning primitives — [`examples/18_ml_primitives.cu`](examples/18_ml_primitives.cu)

The building blocks of neural nets: GEMM, activations, softmax, normalization.

```
   CONCEPTS APPLIED:
     - tiled matmul / cuBLAS (Ch. 11, 20) — the core of every layer
     - reductions for softmax & layernorm (Ch. 07/14): max + sum across a row
     - fused elementwise ops to avoid DRAM round-trips (fusion, Ch. 20)
     - mixed precision / Tensor Cores for the matmuls (Ch. 21)

   SOFTMAX row = exp(x - max(x)) / Σ exp(x - max(x))
     -> a MAX reduction + a SUM reduction + elementwise; fuse to touch DRAM once.
```

---

## 6. Deep learning from scratch — [`examples/21_deep_learning.cu`](examples/21_deep_learning.cu)

A small neural network (forward + backprop) built from CUDA primitives — from
linear regression up toward CNNs.

```
   CONCEPTS APPLIED:
     - GEMM for dense layers (Ch. 11), reductions for losses (Ch. 07)
     - the full training loop: forward -> loss -> backward -> update, all on-GPU
     - streams/graphs to cut per-step overhead (Ch. 13, 16)
     - cuRAND for weight init & dropout (Ch. 20)

   This ties the whole guide together: a real workload is just many fused GEMMs,
   reductions, and elementwise ops, scheduled to keep the GPU busy. Real training
   uses frameworks (PyTorch/JAX on cuDNN/cuBLAS/NCCL) — this shows what's underneath.
```

---

## 7. Also in `examples/`

```
   02_first_kernel.cu           vector add, error checking (Ch. 02)
   03_memory_model.cu           coalescing/constant/bandwidth demos (Ch. 05)
   04_thread_organization.cu    1D/2D/3D indexing (Ch. 03/04)
   05_matrix_operations.cu      naive/tiled/cuBLAS GEMM + transpose (Ch. 11)
   06_shared_memory.cu          reduction/scan/histogram/stencil (Ch. 07)
   07_streams_async.cu          overlap copy+compute (Ch. 13)
   08_advanced_topics.cu        warp prims, atomics, dyn parallelism (Ch. 12/14)
   17_advanced_memory.cu        async copy, vectorized, prefetch (Ch. 15)
   19_multi_gpu.cu              device mgmt, P2P (Ch. 17)
   22_modern_cuda.cu            Tensor Cores / modern features (Ch. 21)
   gpu_locks_and_synchronization.cu   atomics & locks in depth (Ch. 12)
```

---

## 8. A suggested project path

```
   BEGINNER   : vector ops -> image blur -> histogram -> parallel sum reduction
   INTERMEDIATE: tiled matmul -> 2D stencil / heat equation -> bitonic sort ->
                 Mandelbrot renderer
   ADVANCED   : N-body simulation -> BFS/PageRank -> a small MLP with backprop ->
                fused softmax/attention -> multi-GPU data-parallel training
   EXPERT     : a CUTLASS/Tile fused Tensor-Core kernel -> a persistent/graph-based
                inference pipeline -> profile and push a real kernel toward the
                roofline.
```

For each: write it naively, **profile** (Ch. 18), apply the relevant chapter's
optimization, measure again. That loop — measure, optimize, measure — is the whole
job.

---

## 9. Key takeaways

- Every real application is the **fundamentals applied**: 2D indexing +
  tiling (imaging), scan (sorting), shared-memory reuse (N-body/stencils), atomics
  + load balancing (graphs), GEMM + reductions + fusion (ML/DL).
- **Graphs are the hard case** (irregularity/divergence); **GEMM-heavy ML is the
  easy case** (maps beautifully to Tensor Cores).
- For standard pieces, **use libraries** (Ch. 20); hand-write to **fuse** or for
  custom ops.
- Grow via the **project path**, always looping **measure → optimize → measure**
  (Ch. 18) toward the **roofline** (Ch. 00).

**Next:** [99 — Cheatsheet →](99_cheatsheet.md)
