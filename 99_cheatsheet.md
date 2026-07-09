# 99 — CUDA Cheatsheet

> Part of **[CUDA Know-Hows](README.md)**. Everything condensed. Keep this open
> while you code. Links point back to the chapter with the full story.

---

## Kernel & launch ([02](02_first_kernel.md), [03](03_thread_hierarchy.md))

```cpp
__global__ void k(...) { ... }          // GPU kernel, returns void
__device__ float f(...) { ... }         // GPU-only helper
__host__ __device__ int g(...) { }      // compiled for both

// Global 1D index + bounds check:
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) { ... }

// Launch:  <<< grid, block, dynamicSharedBytes, stream >>>
int t = 256, b = (n + t - 1) / t;       // or cuda::ceil_div(n, t)  <cuda/cmath>
k<<<b, t>>>(args);
dim3 block(16,16), grid((W+15)/16, (H+15)/16);   // 2D
k2<<<grid, block>>>(args);

// Grid-stride loop (robust default):
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
    out[i] = ...;
```

Built-ins: `threadIdx`, `blockDim`, `blockIdx`, `gridDim` (`.x/.y/.z`), `warpSize` (32).

---

## Memory ([05](05_memory_model.md), [06](06_memory_management.md))

```cpp
// Explicit:
cudaMalloc(&d, bytes);   cudaFree(d);
cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);       // H2D/D2H/D2D/Default; SYNC
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);   // needs pinned host
cudaMemset(d, 0, bytes);

// Pinned host (fast + required for async copy):
cudaMallocHost(&h, bytes);   cudaFreeHost(h);

// Unified (auto-migrated):
cudaMallocManaged(&p, bytes);
cudaMemPrefetchAsync(p, bytes, device, stream);       // avoid fault migration
cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, device);

// Stream-ordered pool alloc (CUDA 11.2+):
cudaMallocAsync(&p, bytes, stream);   cudaFreeAsync(p, stream);

// On-chip:
__shared__ float tile[256];                 // static shared
extern __shared__ float dyn[];              // dynamic (size = 3rd launch arg)
__constant__ float coeff[64];               // + cudaMemcpyToSymbol(coeff, h, n)
```

```
COALESCING RULE: consecutive threads (threadIdx.x) -> consecutive addresses.
  M[row*W + col] with col = threadIdx.x  ->  coalesced (good)
Vectorized: float4 v = reinterpret_cast<const float4*>(p)[i];  // 16B aligned
```

---

## Shared memory & sync ([07](07_shared_memory.md), [12](12_atomics_and_synchronization.md), [14](14_advanced_kernel_techniques.md))

```cpp
__syncthreads();                            // block barrier (ALL threads, no divergence!)
__syncwarp(mask);                           // warp barrier
tile[ty][tx+1] ...                          // pad [N][N+1] to avoid bank conflicts

// Atomics (global or shared):
atomicAdd(&x, v); atomicMax(&x, v); atomicCAS(&x, old, new); atomicExch(&x, v);

// Warp primitives (mask usually 0xffffffff):
__shfl_down_sync(m, v, delta);  __shfl_sync(m, v, srcLane);  __shfl_xor_sync(m,v,lm);
__ballot_sync(m, pred);  __any_sync(m,pred);  __all_sync(m,pred);  __reduce_add_sync(m,v);

// Warp reduction idiom:
for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffff, val, o);

// Cooperative groups:
namespace cg = cooperative_groups;
auto blk = cg::this_thread_block(); blk.sync();
auto warp = cg::tiled_partition<32>(blk);  int s = cg::reduce(warp, v, cg::plus<int>());
```

---

## Errors, events, streams ([02](02_first_kernel.md), [13](13_streams_and_concurrency.md))

```cpp
#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,\
    cudaGetErrorString(e)); exit(1);} }while(0)
k<<<b,t>>>(...); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());  // launch + exec err

// Events (timing):
cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s); k<<<...>>>(); cudaEventRecord(e); cudaEventSynchronize(e);
float ms; cudaEventElapsedTime(&ms, s, e);

// Streams:
cudaStream_t st; cudaStreamCreate(&st);
k<<<b,t,0,st>>>(...);  cudaMemcpyAsync(...,st);  cudaStreamSynchronize(st);
cudaStreamWaitEvent(st2, ev, 0);            // cross-stream dependency

// Sync scope: cudaEventSynchronize < cudaStreamSynchronize < cudaDeviceSynchronize
```

---

## CUDA Graphs ([16](16_cuda_graphs.md))

```cpp
cudaGraph_t g; cudaGraphExec_t x;
cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  /* issue kernels/copies on st */
cudaStreamEndCapture(st, &g);
cudaGraphInstantiate(&x, g, 0,0,0);         // once (costly)
for (...) cudaGraphLaunch(x, st);           // replay (cheap)
cudaGraphExecUpdate(x, gNew, ...);          // params changed, topology same
```

---

## Compile ([01](01_setup_and_compilation.md))

```bash
nvcc -O3 -std=c++17 -arch=sm_80 prog.cu -o prog          # single arch
nvcc -arch=native prog.cu -o prog                         # this machine's GPU
nvcc -gencode arch=compute_80,code=sm_80 \                # multi-arch + PTX
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 prog.cu -o prog
# useful flags:
-lineinfo            # profiler line info (no deopt)
-g -G                # debug host/device (deoptimizes device)
-Xptxas -v           # print regs/smem per kernel  <-- read this for occupancy
--use_fast_math      # fast, lower-precision math
-maxrregcount=N  __launch_bounds__(T,B)   # register/occupancy control
-rdc=true -lcudadevrt                      # dynamic parallelism / device linking
-lcublas -lcurand -lcufft -lcudnn -lnccl   # libraries
# inspect:
cuobjdump -sass prog     nvcc -ptx prog.cu     godbolt.org
```

Arch flags: Volta `sm_70`, Turing `sm_75`, Ampere `sm_80/86`, Ada `sm_89`,
Hopper `sm_90(a)`, Blackwell `sm_100(a)`/`sm_120`.

---

## Profiling & debugging ([18](18_profiling_and_debugging.md))

```bash
compute-sanitizer ./prog            # memcheck/racecheck/initcheck (like ASan)
nsys profile -o rep ./prog          # Nsight Systems: timeline (find bottlenecks)
ncu --set full -o rep ./prog        # Nsight Compute: per-kernel deep analysis
cuda-gdb ./prog                     # interactive device debugging
nvidia-smi   nvidia-smi topo -m     # GPUs / interconnect topology
```

```
NSIGHT COMPUTE metrics to check: Memory Throughput %, Global Load/Store
Efficiency % (coalescing), Achieved Occupancy, Tensor Core utilization, L2 hit %.
```

---

## Device query

```cpp
cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
// p.multiProcessorCount, p.maxThreadsPerBlock, p.sharedMemPerBlock,
// p.warpSize, p.totalGlobalMem, p.major/.minor (compute capability)
cudaSetDevice(d);  cudaGetDeviceCount(&n);
cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, k, 0, 0);
```

---

## Latency numbers to keep in your head ([05](05_memory_model.md), [08](08_execution_model_and_occupancy.md))

```
register           ~1 cycle          | warp                 32 threads
shared / L1        ~20-30 cycles     | max threads/block    1024
L2                 ~200 cycles       | block size default   128 or 256 (mult of 32)
global (DRAM)      ~400-800 cycles   | HBM3e bandwidth      ~2-8 TB/s
PCIe Gen5          ~32-64 GB/s       | NVLink 5             ~1.8 TB/s
constant (bcast)   ~5 cycles         | FP64 on consumer     1/32-1/64 of FP32
```

---

## The optimization checklist ([08](08_execution_model_and_occupancy.md), [18](18_profiling_and_debugging.md), [19](19_optimization_case_studies.md))

```
1. CORRECT first (compute-sanitizer clean, verified vs CPU).
2. PROFILE (nsys) -> is it kernel-bound, transfer-bound, or launch-bound?
3. MEMORY-BOUND kernel?
   [ ] coalesce global access (threadIdx.x -> contiguous)
   [ ] reuse in shared memory / registers (tiling)
   [ ] vectorized loads (float4); reduce bytes moved
   [ ] fewer/overlapped H2D-D2H transfers (streams, pinned)  [ ] unified prefetch
4. COMPUTE-BOUND kernel?
   [ ] better math mix / --use_fast_math / fmaf   [ ] Tensor Cores (libraries)
   [ ] less redundant work; unroll hot loops
5. UNDER-UTILIZED GPU?
   [ ] enough parallelism/occupancy to hide latency (check -Xptxas -v regs/smem)
   [ ] no warp divergence on hot paths          [ ] fill whole waves / grid-stride
   [ ] launch-bound? -> CUDA Graphs
6. Could a LIBRARY do it faster? (cuBLAS/cuDNN/CUB/CUTLASS)  -> use it.
7. MEASURE AGAIN. Compare to the roofline. Stop when at the relevant peak.
```

← Back to [README](README.md)
