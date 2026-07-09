# 20 — Libraries & Ecosystem

> Part of **[CUDA Know-Hows](README.md)**. Prev: [19 — Optimization case studies](19_optimization_case_studies.md).
> Next: [21 — Modern CUDA](21_modern_cuda.md).
>
> Goal: know what NOT to write yourself. NVIDIA and the community ship tuned,
> battle-tested libraries for linear algebra, FFT, deep learning, parallel
> algorithms, and communication. An expert reaches for these first and hand-writes
> only what they don't cover (usually fusion). This is your map of the ecosystem.

---

## 1. The golden rule of libraries

```
   Before writing a kernel, ask: does a library already do this?
     - Standard ops (GEMM, conv, FFT, sort, scan, reduce) -> almost certainly YES,
       and the library will beat your kernel (Ch. 11 showed cuBLAS at ~90% of peak).
     - Your value-add is usually FUSION: combining ops to avoid DRAM round-trips,
       or custom ops libraries don't have.
   Hand-rolling standard ops is a rite of passage for LEARNING (do it once),
   but a mistake in PRODUCTION.
```

---

## 2. Math libraries (linear algebra, FFT, RNG, sparse)

```
   ┌─────────────┬─────────────────────────────────────────────────────────────────┐
   │ cuBLAS      │ dense BLAS (GEMM, GEMV, ...). The workhorse. cuBLASLt for       │
   │             │ fused epilogues, mixed precision, Tensor Core control.          │
   │ cuBLASDx    │ device-side BLAS you call FROM a kernel (fuse GEMM into yours). │
   │ cuSOLVER    │ dense/sparse LAPACK: LU, QR, Cholesky, SVD, eigen.              │
   │ cuSPARSE    │ sparse matrices: SpMV, SpMM, sparse formats (CSR/COO/...).      │
   │ cuFFT       │ fast Fourier transforms (1D/2D/3D, batched). cuFFTDx device.    │
   │ cuRAND      │ random number generation (host + device APIs).                  │
   │ cuTENSOR    │ tensor contractions / permutations (Einstein-summation style).  │
   │ AmgX        │ algebraic multigrid solvers for large sparse systems.           │
   └─────────────┴─────────────────────────────────────────────────────────────────┘
```

```cpp
// cuBLAS GEMM (Ch. 11): C = alpha*A*B + beta*C
cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);

// cuFFT: forward complex-to-complex FFT
cufftHandle plan; cufftPlan1d(&plan, N, CUFFT_C2C, 1);
cufftExecC2C(plan, dIn, dOut, CUFFT_FORWARD);

// cuRAND: fill device array with uniform randoms
curandGenerator_t g; curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
curandGenerateUniform(g, dData, N);
```

---

## 3. Deep learning libraries

```
   ┌─────────────┬────────────────────────────────────────────────────────────────┐
   │ cuDNN       │ deep-learning primitives: convolution, pooling, normalization, │
   │             │ activation, attention. What PyTorch/TF call under the hood.    │
   │ CUTLASS     │ open-source C++ templates for GEMM/conv at cuBLAS-class speed; │
   │             │ the go-to for CUSTOM fused Tensor Core kernels. CuTe = its     │
   │             │ layout/tensor abstraction. (Ch. 21 for Tensor Cores.)          │
   │ TensorRT    │ inference optimizer/runtime: fuses layers, picks kernels,      │
   │             │ quantizes (INT8/FP8/FP4), builds CUDA graphs. Deployment.      │
   │ cuDNN       │ + Frontend API for graph-based op fusion.                      │
   │ Transformer │ NVIDIA Transformer Engine: FP8 training/inference for LLMs.    │
   │  Engine     │                                                                │
   └─────────────┴────────────────────────────────────────────────────────────────┘
```

```
   WHEN TO USE WHAT FOR DL:
     training a model         -> a framework (PyTorch/JAX) on cuDNN/cuBLAS + NCCL
     a custom fused op        -> CUTLASS (Tensor Core GEMM/conv) or a Triton kernel
     deploying for inference  -> TensorRT (or TensorRT-LLM for large language models)
     FP8 LLM training         -> Transformer Engine
```

---

## 4. Parallel algorithms & C++ core libraries (CCCL)

The **CUDA Core Compute Libraries (CCCL)** unify three C++ libraries you'll use
constantly. Prefer these over hand-rolled reduce/scan/sort (Ch. 07 taught the
patterns; these are the production versions).

```
   ┌─────────────┬────────────────────────────────────────────────────────────────┐
   │ Thrust      │ STL-like host API: thrust::sort, reduce, transform, scan on    │
   │             │ device_vector. Fast to write, great for prototypes & glue.     │
   │ CUB         │ lower-level, tuned building blocks: DeviceReduce, DeviceScan,  │
   │             │ DeviceRadixSort, plus BLOCK/WARP-level primitives for YOUR     │
   │             │ kernels (BlockReduce, BlockScan, WarpReduce).                  │
   │ libcu++     │ the CUDA C++ standard library: cuda::std::atomic, <cuda/cmath> │
   │  (<cuda/*>) │ (ceil_div), cuda::pipeline, cuda::barrier, cuda::memcpy_async. │
   └─────────────┴────────────────────────────────────────────────────────────────┘
```

```cpp
// Thrust: sort a device vector in one line
thrust::device_vector<int> d = h;
thrust::sort(d.begin(), d.end());

// CUB: device-wide reduction (allocates temp storage in two-call idiom)
size_t tmpBytes = 0;
cub::DeviceReduce::Sum(nullptr, tmpBytes, dIn, dOut, N);   // query size
void* tmp; cudaMalloc(&tmp, tmpBytes);
cub::DeviceReduce::Sum(tmp, tmpBytes, dIn, dOut, N);       // run

// CUB in YOUR kernel: block-wide reduction primitive
using BR = cub::BlockReduce<float, 256>;
__shared__ typename BR::TempStorage tmp;
float total = BR(tmp).Sum(myVal);   // correct, tuned block reduce (Ch. 07/14)
```

---

## 5. Communication & multi-node

```
   NCCL   : GPU collective communication (AllReduce, etc.) — the DL backbone (Ch. 17).
   NVSHMEM: partitioned global address space (PGAS) across GPUs — fine-grained,
            one-sided GPU-initiated communication for tightly-coupled algorithms.
   MPI    : classic distributed computing; combine with CUDA (CUDA-aware MPI passes
            device pointers directly). See cpp-hpc Module 12 for MPI fundamentals.
   GPUDirect RDMA: NICs read/write GPU memory directly, bypassing the CPU, for
            multi-node scaling.
```

---

## 6. Languages & higher-level entry points

```
   Triton   : Python DSL for GPU kernels; you write tile-level Python, it compiles
              to fast PTX. Hugely popular for custom DL kernels (fused attention,
              quantized matmuls). Great productivity/perf trade-off.
   CUDA Python / Numba: write kernels in Python (cuda.jit) or drive CUDA from
              Python (cuda.core, cupy). CuPy = NumPy on the GPU.
   OpenACC / OpenMP target: directive-based offload for existing C/C++/Fortran.
   std::par : NVC++ can run C++ standard parallel algorithms on the GPU.
   Kokkos / RAJA / SYCL: performance-portable C++ that also target non-NVIDIA HW.
   CUDA Tile C++: NVIDIA's tile-level abstraction (Ch. 21).
```

---

## 7. How to choose (decision guide)

```
   Standard dense linear algebra      -> cuBLAS / cuBLASLt / CUTLASS
   FFT / signal processing            -> cuFFT
   Sparse / solvers                   -> cuSPARSE / cuSOLVER / AmgX
   sort / scan / reduce / select      -> Thrust (easy) or CUB (fused/tuned)
   deep learning (framework)          -> PyTorch/JAX (cuDNN+cuBLAS+NCCL) 
   custom fused DL kernel             -> CUTLASS or Triton
   inference deployment               -> TensorRT / TensorRT-LLM
   multi-GPU communication            -> NCCL (or NVSHMEM for fine-grained)
   quick Python prototype             -> CuPy / Numba / Triton
   only hand-write a raw kernel when  -> no library covers it, OR you need FUSION
                                         to avoid DRAM round-trips between library
                                         calls.
```

Several `examples/` use these (`05_matrix_operations.cu` → cuBLAS,
`14_scientific_computing.cu`/`18_ml_primitives.cu` → cuRAND/cuBLAS).

---

## 8. Key takeaways

- **Don't reinvent standard ops** — cuBLAS/cuDNN/cuFFT/CUTLASS/CUB beat hand-rolled
  kernels; your job is usually **fusion** and custom ops.
- **Math**: cuBLAS(Lt), cuSOLVER, cuSPARSE, cuFFT, cuRAND, cuTENSOR.
- **Deep learning**: cuDNN (primitives), CUTLASS (custom Tensor Core kernels),
  TensorRT (inference), Transformer Engine (FP8).
- **Parallel algorithms** via **CCCL**: Thrust (easy), CUB (tuned + block/warp
  primitives), libcu++ (`cuda::` std types, pipelines, `ceil_div`).
- **Communication**: NCCL (collectives), NVSHMEM, CUDA-aware MPI, GPUDirect RDMA.
- Higher-level: **Triton**, CuPy/Numba, std::par, Kokkos/RAJA/SYCL, CUDA Tile.

**Next:** [21 — Modern CUDA →](21_modern_cuda.md)
