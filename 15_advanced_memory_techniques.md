# 15 — Advanced Memory Techniques

> Part of **[CUDA Know-Hows](README.md)**. Prev: [14 — Advanced kernel techniques](14_advanced_kernel_techniques.md).
> Next: [16 — CUDA graphs](16_cuda_graphs.md).
> Runnable code: [`examples/17_advanced_memory.cu`](examples/17_advanced_memory.cu).
>
> Goal: the memory optimizations used in production kernels — asynchronous shared-
> memory copies (`cp.async`), the Tensor Memory Accelerator (TMA), software
> pipelining, L2 residency control, vectorized loads, and unified-memory tuning.
> These are what let modern GEMM/attention kernels hit peak.

---

## 1. The problem these solve: hiding global→shared latency

Classic tiling (Ch. 07/11) does: load tile to shared → `__syncthreads()` →
compute → repeat. But the load uses registers and *stalls* until data arrives.
Modern hardware can copy global→shared **asynchronously**, so a warp issues the
next tile's copy and computes on the current tile while it flies in — software
pipelining / double buffering.

```
   SYNCHRONOUS tiling (each step stalls on the load):
     [load tile0 ■■stall■■][compute0][load tile1 ■■stall■■][compute1] ...

   ASYNC / DOUBLE-BUFFERED (copy overlaps compute):
     copy:    [cp tile0][cp tile1][cp tile2]
     compute:          [compute0 ][compute1 ][compute2]
                        └ compute tile0 WHILE tile1 copies in -> no stall
```

---

## 2. Asynchronous copy: `cp.async` / `memcpy_async` (Ampere sm_80+)

`cp.async` copies global→shared **without going through registers** and without
blocking the thread. The CUDA C++ way is `cooperative_groups::memcpy_async` +
a pipeline, or `cuda::memcpy_async` from libcu++.

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__global__ void k(const float* g) {
    __shared__ float smem[256];
    cg::thread_block block = cg::this_thread_block();
    // async copy 256 floats global->shared, bypassing registers:
    cg::memcpy_async(block, smem, g, sizeof(float)*256);
    cg::wait(block);                    // wait for the copy to complete
    block.sync();
    // ... use smem ...
}
```

```
   cp.async benefits (Ampere+):
     - global -> shared WITHOUT occupying registers (frees them for compute)
     - non-blocking: issue copy, do other work, then wait
     - enables true double-buffered pipelines (below)
   Pre-Ampere, the "async copy" is emulated (a normal load) — no benefit.
```

---

## 3. Software pipelining with `cuda::pipeline`

libcu++ provides a `pipeline` object to manage multi-stage double/triple
buffering cleanly: commit a stage's async copies, and consume completed stages.

```cpp
#include <cuda/pipeline>
// Sketch: 2-stage pipeline over K tiles
auto pipe = cuda::make_pipeline();
for (int t = 0; t < numTiles; ++t) {
    pipe.producer_acquire();
    cuda::memcpy_async(smem[t%2], gmem + t*TILE, shape, pipe);  // stage the copy
    pipe.producer_commit();
    if (t > 0) {
        pipe.consumer_wait();          // wait for the PREVIOUS tile's copy
        compute(smem[(t-1)%2]);         // compute on it while THIS tile copies
        pipe.consumer_release();
    }
}
```

```
   TWO-STAGE PIPELINE (ping-pong buffers in shared memory):
     buffer A: [copy t0]        [compute t0]        [copy t2] ...
     buffer B:          [copy t1]        [compute t1]        ...
   While one buffer is computed, the other is being filled. This is the core of
   fast GEMM/attention kernels. CUTLASS generalizes it to N-stage pipelines.
```

---

## 4. Tensor Memory Accelerator (TMA) — Hopper sm_90+

TMA is a dedicated hardware unit for **bulk, asynchronous, multi-dimensional**
copies between global and shared memory. A single thread issues a descriptor and
the hardware handles all the address generation and bounds — freeing CUDA cores
entirely from copy bookkeeping.

```
   WITHOUT TMA: every thread computes addresses, issues loads, handles edges.
   WITH TMA:    one thread kicks off a whole tile copy via a descriptor:

     [tensor map descriptor] --▶ TMA engine copies a 2D/3D tile global<->shared
                                 with bounds handling, async, no core involvement

   + massively reduces address-compute instructions and register pressure
   + multi-dimensional (tiles of matrices) with built-in boundary handling
   + pairs with async barriers (mbarrier) to signal completion
   Used by cuBLAS/CUTLASS Hopper GEMM and FlashAttention-class kernels.
```

You rarely write TMA by hand; you use it via CUTLASS or the CUDA Tile API
(Ch. 21). Know it exists and why it matters: it's a big reason Hopper/Blackwell
GEMM is so fast.

---

## 5. L2 cache residency control (Ampere sm_80+)

You can hint that a region of global memory should be kept "persistent" in L2 —
useful when a kernel repeatedly reads the same data (e.g. weights, a lookup
table) that fits in a slice of L2.

```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = dHotData;
attr.accessPolicyWindow.num_bytes = hotBytes;              // <= carveout size
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

```
   L2 PERSISTENCE: reserve part of L2 for hot, reused data so it isn't evicted by
   streaming traffic. Wins when the reused footprint fits the carveout. Overuse
   starves normal caching — measure L2 hit rate in Nsight (Ch. 18).
```

---

## 6. Vectorized memory access (free bandwidth)

Loading 128 bits per thread in one instruction (`float4`, `int4`) uses fewer,
wider transactions and reduces instruction count — a cheap, portable win when
data is 16-byte aligned.

```cpp
// Process 4 elements per thread with one 128-bit load/store:
float4 a = reinterpret_cast<const float4*>(in)[i];
float4 b = reinterpret_cast<const float4*>(y)[i];
a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
reinterpret_cast<float4*>(out)[i] = a;
```

```
   float4 load = one 16-byte transaction per thread instead of four 4-byte ones.
   Requires 16-byte alignment (cudaMalloc returns >=256-byte aligned, so array
   starts are fine; watch offsets). Great for copy/elementwise/SAXPY-type kernels.
```

---

## 7. Unified memory performance tuning

Unified memory (Ch. 02/06) is convenient but naive use causes page-fault
migrations at access time. Prefetch and advise to get explicit-copy-like
performance:

```cpp
cudaMemPrefetchAsync(ptr, bytes, deviceId, stream);   // move pages to GPU ahead of use
cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, deviceId);       // duplicate read-only
cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, deviceId);// pin location
```

```
   NAIVE unified mem:  first GPU access -> PAGE FAULT -> migrate page -> stall
   TUNED unified mem:  cudaMemPrefetchAsync before the kernel -> data already there
   Advise hints reduce thrashing for read-mostly or shared data. Profile page-fault
   counts in Nsight Systems.
```

---

## 8. Zero-copy & mapped memory (use sparingly)

Pinned host memory can be *mapped* into the GPU address space so kernels read it
directly over PCIe (no explicit copy). Good only for small, infrequently accessed
data or when the access is truly sparse — otherwise the per-access PCIe latency
kills performance.

```cpp
cudaHostAlloc(&h, bytes, cudaHostAllocMapped);
float* dPtr; cudaHostGetDevicePointer(&dPtr, h, 0);   // GPU reads host memory directly
```

Run the example to see async copy / vectorized / prefetch effects:

```bash
cd examples && make 17_advanced_memory && ./17_advanced_memory
```

---

## 9. Key takeaways

- Hide global→shared latency with **asynchronous copies** (`cp.async` /
  `cg::memcpy_async`, Ampere+): copy bypasses registers and overlaps compute.
- Build **software pipelines / double buffering** (`cuda::pipeline`) so the next
  tile copies while the current one computes — the heart of fast GEMM/attention.
- **TMA** (Hopper+) offloads bulk multi-dim copies to hardware; you get it via
  CUTLASS / CUDA Tile.
- Control **L2 residency** for hot reused data; use **vectorized loads** (`float4`)
  for free bandwidth when aligned.
- Tune **unified memory** with `cudaMemPrefetchAsync` / `cudaMemAdvise` to avoid
  fault-driven migration; use **zero-copy/mapped** memory only for small/sparse
  access.

**Next:** [16 — CUDA graphs →](16_cuda_graphs.md)
