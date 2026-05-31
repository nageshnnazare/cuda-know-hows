/*
 * CUDA Tutorial - Part 22: Modern CUDA Programming (2024-2026)
 *
 * This file covers the latest CUDA features and best practices:
 * 1. CUDA Graphs (capture, replay, conditional nodes)
 * 2. Cooperative Groups (flexible thread synchronization)
 * 3. Mixed-Precision Programming (FP16, BF16, TF32, FP8, FP4)
 * 4. Stream-Ordered Memory Allocation (cudaMallocAsync)
 * 5. Architecture-Aware Programming (Hopper & Blackwell)
 * 6. CUTLASS & CuTe Overview (high-performance GEMM templates)
 * 7. CUDA Tile Programming Model (CUDA 13.1+)
 * 8. Performance Best Practices for Modern GPUs
 *
 * Compile: nvcc -std=c++17 -arch=sm_80 -o modern_cuda 22_modern_cuda.cu
 * Run:     ./modern_cuda
 *
 * For CUDA Tile examples, compile with:
 *   nvcc -std=c++20 --enable-tile -arch=sm_80 -o tile_example tile_example.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * ═══════════════════════════════════════════════════════════════════
 *          SECTION 1: CUDA GRAPHS - REDUCING LAUNCH OVERHEAD
 * ═══════════════════════════════════════════════════════════════════
 *
 * CUDA Graphs capture a sequence of operations (kernels, memory copies)
 * into a graph that can be launched as a single unit. This dramatically
 * reduces CPU launch overhead, especially for workloads with many small
 * kernels (common in deep learning inference).
 *
 * Traditional approach:
 *   CPU: [launch K1][launch K2][launch K3][launch K4]  ← overhead per launch
 *   GPU:     [K1]       [K2]       [K3]       [K4]
 *
 * With CUDA Graphs:
 *   CPU: [launch graph]                                 ← single launch
 *   GPU:     [K1][K2][K3][K4]                           ← no gaps
 *
 * CUDA 12.8+ adds conditional graph nodes:
 *   - IF/ELSE: execute one of two subgraphs based on a device-side condition
 *   - SWITCH: select from multiple subgraphs
 *   - WHILE: loop a subgraph until a condition is false
 *
 * These enable dynamic control flow INSIDE a graph without CPU intervention.
 */

__global__ void scaleKernel(float *data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scalar;
    }
}

__global__ void biasKernel(float *data, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += bias;
    }
}

__global__ void reluKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

void demonstrateCudaGraphs() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 1: CUDA Graphs\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i - N/2) / N;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    /*
     * Stream Capture: record a sequence of operations into a graph
     *
     *   begin capture → launch kernels → end capture → instantiate → launch
     *
     * The captured graph can be launched thousands of times with minimal
     * CPU overhead (typically <5 microseconds per launch).
     */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    scaleKernel<<<gridSize, blockSize, 0, stream>>>(d_data, 2.0f, N);
    biasKernel<<<gridSize, blockSize, 0, stream>>>(d_data, 1.0f, N);
    reluKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*
     * Compare: individual kernel launches vs graph launch
     */

    // Warm up
    for (int i = 0; i < 10; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Time graph launches
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 1000; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float graphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graphTime, start, stop));

    // Time individual kernel launches
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 1000; i++) {
        scaleKernel<<<gridSize, blockSize, 0, stream>>>(d_data, 2.0f, N);
        biasKernel<<<gridSize, blockSize, 0, stream>>>(d_data, 1.0f, N);
        reluKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float individualTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&individualTime, start, stop));

    printf("1000 iterations of 3-kernel pipeline:\n");
    printf("  Individual launches: %.3f ms\n", individualTime);
    printf("  CUDA Graph:          %.3f ms\n", graphTime);
    printf("  Speedup:             %.2fx\n\n", individualTime / graphTime);

    printf("When to use CUDA Graphs:\n");
    printf("  - Repeated execution of the same kernel sequence\n");
    printf("  - Many small kernels (inference pipelines)\n");
    printf("  - Low-latency requirements\n");
    printf("  - CUDA 12.8+: use IF/ELSE/SWITCH nodes for dynamic control\n\n");

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *      SECTION 2: COOPERATIVE GROUPS - FLEXIBLE SYNCHRONIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Cooperative Groups (CUDA 9+, enhanced through 13.x) provides a
 * flexible API for thread synchronization at any granularity:
 *
 *   Grid Level  ← All blocks cooperate (requires cooperative launch)
 *   Block Level ← Traditional __syncthreads() equivalent
 *   Tile Level  ← Sub-warp groups (e.g., 4, 8, 16 threads)
 *   Warp Level  ← Full warp (32 threads)
 *   Coalesced   ← Only the active/converged threads
 *
 * Key benefits over raw __syncthreads():
 *   - Type-safe: the group type encodes what sync is legal
 *   - Composable: partition groups into sub-groups
 *   - Portable: works across architectures
 */

__global__ void cooperativeGroupsReduction(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    /*
     * Warp-level reduction using cooperative groups shuffle.
     * This replaces the older __shfl_down_sync() pattern with
     * a type-safe, group-aware version.
     *
     * Before (manual warp sync):
     *   val += __shfl_down_sync(0xFFFFFFFF, val, 16);
     *   val += __shfl_down_sync(0xFFFFFFFF, val, 8);
     *   ...
     *
     * After (cooperative groups):
     *   for (int offset = warp.size()/2; offset > 0; offset /= 2)
     *       val += warp.shfl_down(val, offset);
     */
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    __shared__ float warpSums[32];

    int laneId = warp.thread_rank();
    int warpId = threadIdx.x / warp.size();

    if (laneId == 0) {
        warpSums[warpId] = val;
    }

    block.sync();

    if (warpId == 0) {
        val = (laneId < blockDim.x / warp.size()) ? warpSums[laneId] : 0.0f;
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            val += warp.shfl_down(val, offset);
        }
        if (laneId == 0) {
            atomicAdd(output, val);
        }
    }
}

/*
 * Tiled partition example: sub-warp groups for fine-grained parallelism.
 * Useful for problems where 32-wide SIMD is too coarse.
 */
__global__ void tiledPartitionExample(float *data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];

    // 8-thread reduction: each group of 8 threads computes a partial sum
    for (int offset = tile8.size() / 2; offset > 0; offset /= 2) {
        val += tile8.shfl_down(val, offset);
    }

    if (tile8.thread_rank() == 0) {
        data[idx] = val;
    }
}

void demonstrateCooperativeGroups() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 2: Cooperative Groups\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cooperativeGroupsReduction<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Cooperative Groups reduction of %d ones: %.0f (expected: %d)\n", N, result, N);
    printf("Result: %s\n\n",
           (fabs(result - (float)N) < N * 0.001f) ? "PASSED" : "FAILED");

    printf("Cooperative Groups hierarchy:\n");
    printf("  grid_group         ← All blocks (cooperative launch)\n");
    printf("  multi_grid_group   ← Multiple GPUs (deprecated, use NCCL)\n");
    printf("  thread_block       ← __syncthreads() equivalent\n");
    printf("  thread_block_tile  ← Sub-warp: 1,2,4,8,16,32 threads\n");
    printf("  coalesced_threads  ← Only converged threads in a warp\n\n");

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    free(h_data);
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *     SECTION 3: MIXED-PRECISION PROGRAMMING & DATA TYPES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Modern GPUs have dedicated hardware for reduced-precision arithmetic.
 * Using lower precision where acceptable can yield massive speedups.
 *
 * ┌─────────┬───────┬────────────┬──────────────────────────────────┐
 * │ Type    │ Bits  │ Range      │ When to Use                      │
 * ├─────────┼───────┼────────────┼──────────────────────────────────┤
 * │ FP64    │ 64    │ ±10^308    │ Scientific computing, finance    │
 * │ FP32    │ 32    │ ±10^38     │ Default, general purpose         │
 * │ TF32    │ 19*   │ ±10^38     │ Training (Ampere+, via TC)       │
 * │ FP16    │ 16    │ ±65504     │ Inference, training with scaling │
 * │ BF16    │ 16    │ ±10^38     │ Training (same range as FP32)    │
 * │ FP8 e4m3│ 8     │ ±240       │ Inference, fine-tuning (Hopper+) │
 * │ FP8 e5m2│ 8     │ ±57344     │ Gradients (wider range)          │
 * │ FP6     │ 6     │ varies     │ Inference (Blackwell+)           │
 * │ FP4     │ 4     │ ±6         │ Inference (Blackwell+, block-    │
 * │         │       │            │ scaled with shared exponents)    │
 * └─────────┴───────┴────────────┴──────────────────────────────────┘
 * *TF32 has 10-bit mantissa + 8-bit exponent, used transparently by TC
 *
 * Block-Scaled Formats (Blackwell):
 *   MXFP4: Each group of 32 elements shares an 8-bit scale factor.
 *           Hardware performs rescaling automatically in the tensor core.
 *           Doubles throughput vs FP8 with minimal quality loss for
 *           inference and (experimentally) training.
 *
 *   NVFP4: NVIDIA's proprietary format, similar structure.
 *
 * Performance relative to FP32 on tensor cores:
 *   FP64:  1x  ← Baseline (only datacenter GPUs have good FP64)
 *   FP32:  1x  ← CUDA core baseline
 *   TF32:  ~8x ← Tensor core, transparent on Ampere+
 *   FP16:  ~16x
 *   BF16:  ~16x
 *   FP8:   ~32x  (Hopper+)
 *   FP4:   ~64x  (Blackwell+)
 *
 * Strategy: accumulate in higher precision (FP32), compute in lower precision.
 * This is exactly what Tensor Cores do: multiply in FP16/BF16/FP8/FP4,
 * accumulate in FP32.
 */

__global__ void mixedPrecisionDemo(const half *a, const half *b,
                                   float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        /*
         * half (FP16) arithmetic: 2x memory savings, hardware-accelerated.
         * Use __hadd, __hmul for FP16 operations.
         * Use __float2half / __half2float for conversion.
         *
         * For best performance, use half2 (vectorized 2-wide FP16):
         *   half2 a2 = *(half2*)&a[idx*2];
         *   half2 b2 = *(half2*)&b[idx*2];
         *   half2 c2 = __hadd2(a2, b2);
         */
        float fa = __half2float(a[idx]);
        float fb = __half2float(b[idx]);
        c[idx] = fa * fb;
    }
}

void demonstrateMixedPrecision() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 3: Mixed-Precision Data Types\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    const int N = 1 << 20;

    half *h_a = (half*)malloc(N * sizeof(half));
    half *h_b = (half*)malloc(N * sizeof(half));
    float *h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half((float)i / N);
        h_b[i] = __float2half(2.0f);
    }

    half *d_a, *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(half), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    mixedPrecisionDemo<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("FP16 input → FP32 accumulation:\n");
    printf("  %d elements processed in %.3f ms\n", N, ms);
    printf("  Memory saved: 50%% (2 bytes vs 4 bytes per element)\n\n");

    printf("Precision selection guide:\n");
    printf("  ┌────────────────────────────────────────────────────────────┐\n");
    printf("  │ Use Case              │ Recommended Precision              │\n");
    printf("  ├────────────────────────────────────────────────────────────┤\n");
    printf("  │ LLM Training          │ BF16 compute, FP32 accumulate      │\n");
    printf("  │ LLM Inference         │ FP8 or FP4 (Hopper/Blackwell)      │\n");
    printf("  │ CV Training           │ FP16 or BF16 with loss scaling     │\n");
    printf("  │ CV Inference          │ FP16 or INT8                       │\n");
    printf("  │ Scientific Computing  │ FP64 or FP32                       │\n");
    printf("  │ General GPU Computing │ FP32 (safe default)                │\n");
    printf("  └────────────────────────────────────────────────────────────┘\n\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *     SECTION 4: STREAM-ORDERED MEMORY ALLOCATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Traditional cudaMalloc/cudaFree are synchronous and global operations.
 * Stream-ordered allocation (CUDA 11.2+) ties memory lifetime to a stream,
 * enabling better overlap and reducing fragmentation.
 *
 * Traditional:
 *   cudaMalloc(&ptr, size);    ← synchronizes all streams
 *   kernel<<<..., stream>>>();
 *   cudaFree(ptr);             ← synchronizes all streams
 *
 * Stream-ordered:
 *   cudaMallocAsync(&ptr, size, stream);  ← non-blocking
 *   kernel<<<..., stream>>>();
 *   cudaFreeAsync(ptr, stream);           ← non-blocking
 *
 * Benefits:
 *   - No implicit synchronization
 *   - Memory pools reduce OS allocation overhead
 *   - Better for workloads with dynamic memory needs (GNNs, sparse ops)
 */

void demonstrateStreamOrderedAlloc() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 4: Stream-Ordered Memory Allocation\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Traditional allocation
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        float *d_tmp;
        CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
        scaleKernel<<<(N+255)/256, 256, 0, stream>>>(d_tmp, 1.0f, N);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_tmp));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float traditionalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&traditionalTime, start, stop));

    // Stream-ordered allocation
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        float *d_tmp;
        CUDA_CHECK(cudaMallocAsync(&d_tmp, bytes, stream));
        scaleKernel<<<(N+255)/256, 256, 0, stream>>>(d_tmp, 1.0f, N);
        CUDA_CHECK(cudaFreeAsync(d_tmp, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float asyncTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&asyncTime, start, stop));

    printf("100 alloc-compute-free cycles:\n");
    printf("  cudaMalloc/cudaFree:         %.3f ms\n", traditionalTime);
    printf("  cudaMallocAsync/cudaFreeAsync: %.3f ms\n", asyncTime);
    printf("  Speedup:                     %.2fx\n\n", traditionalTime / asyncTime);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *    SECTION 5: ARCHITECTURE-AWARE PROGRAMMING PATTERNS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Modern CUDA code should be aware of the target architecture to
 * leverage specialized features while maintaining portability.
 *
 * Compile-time architecture detection:
 *   #if __CUDA_ARCH__ >= 900   // Hopper
 *   #if __CUDA_ARCH__ >= 1000  // Blackwell datacenter
 *   #if __CUDA_ARCH__ >= 1200  // Blackwell consumer
 *
 * Runtime detection:
 *   cudaDeviceProp::major/minor
 *
 * Key per-architecture features:
 *
 * ┌─────────────┬──────┬────────────────────────────────────────┐
 * │ Feature     │ Arch │ What It Enables                        │
 * ├─────────────┼──────┼────────────────────────────────────────┤
 * │ Async copy  │ 8.0+ │ cp.async: overlap compute+data move    │
 * │ TMA         │ 9.0+ │ HW address gen for bulk async copies   │
 * │ WGMMA       │ 9.0+ │ 128-thread warp-group matrix ops       │
 * │ Clusters    │ 9.0+ │ Block cooperation via dist. shared mem │
 * │ DPX         │ 9.0+ │ HW dynamic programming instructions    │
 * │ tcgen05.mma │10.0+ │ Single-thread tensor core launch       │
 * │ TMEM        │10.0+ │ Dedicated tensor memory                │
 * │ FP4/FP6     │10.0+ │ Native 4/6-bit tensor core support     │
 * │ CTA pairs   │10.0+ │ Two CTAs share operands in a TPC       │
 * │ Block-scale │12.0+ │ mma.sync with block scaling (consumer) │
 * └─────────────┴──────┴────────────────────────────────────────┘
 */

__global__ void archAwareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

#if __CUDA_ARCH__ >= 800
    /*
     * Ampere and later: use async copy for overlapping data movement
     * with computation. On older architectures, fall back to normal loads.
     *
     * In practice, for shared memory staging:
     *   __pipeline_memcpy_async(&smem[tid], &gmem[idx], sizeof(float));
     *   __pipeline_commit();
     *   __pipeline_wait_prior(0);
     *
     * Hopper (9.0+) replaces this with TMA hardware.
     */
    data[idx] = __fmaf_rn(data[idx], 2.0f, 1.0f);
#else
    data[idx] = data[idx] * 2.0f + 1.0f;
#endif
}

void demonstrateArchAwareness() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 5: Architecture-Aware Programming\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int smArch = prop.major * 10 + prop.minor;

    printf("GPU: %s (sm_%d%d = compute capability %d.%d)\n\n",
           prop.name, prop.major, prop.minor, prop.major, prop.minor);

    printf("Feature support for this GPU:\n");
    printf("  Tensor Cores:              %s\n",
           (smArch >= 70) ? "Yes" : "No");
    printf("  FP16 Tensor Core:          %s\n",
           (smArch >= 70) ? "Yes" : "No");
    printf("  BF16 Tensor Core:          %s\n",
           (smArch >= 80) ? "Yes" : "No");
    printf("  TF32 Tensor Core:          %s\n",
           (smArch >= 80) ? "Yes" : "No");
    printf("  Async Copy (cp.async):     %s\n",
           (smArch >= 80) ? "Yes" : "No");
    printf("  FP8 Tensor Core:           %s\n",
           (smArch >= 89) ? "Yes" : "No");
    printf("  TMA (Tensor Memory Accel): %s\n",
           (smArch >= 90) ? "Yes" : "No");
    printf("  Thread Block Clusters:     %s\n",
           (smArch >= 90) ? "Yes" : "No");
    printf("  WGMMA (Warp-Group MMA):    %s\n",
           (smArch >= 90) ? "Yes" : "No");
    printf("  FP4/FP6 Tensor Core:       %s\n",
           (prop.major >= 10) ? "Yes (Blackwell)" : "No");
    printf("  Tensor Memory (TMEM):      %s\n",
           (prop.major >= 10) ? "Yes (Blackwell DC)" : "No");
    printf("  CUDA Tile C++ support:     %s\n",
           (smArch >= 80) ? "Yes (CUDA 13.3+)" : "No");
    printf("\n");

    printf("Architecture-specific best practices:\n");
    printf("  ┌────────────┬────────────────────────────────────────────────┐\n");
    printf("  │ Ampere     │ Use TF32 for matmul, async copy for staging   │\n");
    printf("  │ (sm_80)    │ L2 persistence for working sets < 40MB        │\n");
    printf("  ├────────────┼────────────────────────────────────────────────┤\n");
    printf("  │ Ada        │ FP8 for inference, SER for ray tracing        │\n");
    printf("  │ (sm_89)    │ Thread Block Reconfiguration                  │\n");
    printf("  ├────────────┼────────────────────────────────────────────────┤\n");
    printf("  │ Hopper     │ TMA for data movement, WGMMA for matmul      │\n");
    printf("  │ (sm_90)    │ Clusters for multi-block cooperation          │\n");
    printf("  │            │ FP8 training, warp specialization             │\n");
    printf("  ├────────────┼────────────────────────────────────────────────┤\n");
    printf("  │ Blackwell  │ FP4/FP6 for inference, TMEM for tensor data  │\n");
    printf("  │ (sm_100)   │ tcgen05 single-thread MMA, CTA pair exec     │\n");
    printf("  │            │ 8 TB/s HBM3e, NVLink 5 at 1.8 TB/s          │\n");
    printf("  └────────────┴────────────────────────────────────────────────┘\n\n");
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *     SECTION 6: CUTLASS & CuTe - HIGH-PERFORMANCE GEMM
 * ═══════════════════════════════════════════════════════════════════
 *
 * CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's
 * open-source C++ template library for writing high-performance GEMM
 * kernels. It achieves 95-98% of peak tensor core performance.
 *
 * When to use what:
 *
 * ┌──────────────┬──────────────────────────────────────────────┐
 * │ cuBLAS       │ Standard GEMM shapes, stable API, fastest    │
 * │              │ for standard dtypes. Start here.             │
 * ├──────────────┼──────────────────────────────────────────────┤
 * │ CUTLASS      │ Custom epilogues, unusual dtypes (FP8/FP4),  │
 * │              │ fused operations, sparse GEMM, grouped GEMM  │
 * ├──────────────┼──────────────────────────────────────────────┤
 * │ CuTe         │ CUTLASS's layout algebra DSL; used for       │
 * │              │ building custom tile-based kernels with      │
 * │              │ explicit control over data movement          │
 * ├──────────────┼──────────────────────────────────────────────┤
 * │ CUDA Tile    │ Highest-level: compiler auto-maps to tensor  │
 * │ (CUDA 13.1+) │ cores, shared memory, TMA. Easiest to use.   │
 * ├──────────────┼──────────────────────────────────────────────┤
 * │ Triton       │ Python-native GPU kernel language. Good for  │
 * │              │ prototyping custom kernels.                  │
 * └──────────────┴──────────────────────────────────────────────┘
 *
 * CUTLASS 4.x adds a Python DSL for JIT kernel generation.
 * Supports FP4-FP64 across Volta through Blackwell.
 *
 * Example CUTLASS kernel hierarchy (conceptual):
 *
 *   Device Level → Kernel launch, problem decomposition
 *     Collective Level → Producer/consumer warp specialization
 *       MMA Level → Tensor Core instructions (HMMA, WGMMA, tcgen05)
 *         Copy Level → TMA, cp.async, or global loads
 *
 * For Blackwell specifically:
 *   - tcgen05.mma replaces wgmma as the primary MMA instruction
 *   - Tensor Memory (TMEM) replaces shared memory for accumulation
 *   - CTA pairs share operands through intra-TPC communication
 *   - Block-scaled formats (MXFP4, MXFP8) are first-class citizens
 */

void describeCutlassAndCute() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 6: CUTLASS & CuTe Overview\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("CUTLASS is a header-only C++ library for high-performance GEMM.\n");
    printf("It supports all precisions from FP4 to FP64 across all modern\n");
    printf("NVIDIA architectures (Volta through Blackwell).\n\n");

    printf("Supported GEMM precisions:\n");
    printf("  FP64, FP32, TF32, FP16, BF16, FP8 (e4m3, e5m2),\n");
    printf("  FP6, FP4 (NVFP4, MXFP4), INT8, INT4, Binary\n\n");

    printf("Key CUTLASS concepts:\n");
    printf("  1. Tile: A sub-problem processed by one thread block\n");
    printf("  2. Warp Specialization: Producer warps load data,\n");
    printf("     consumer warps compute MMA operations\n");
    printf("  3. Software Pipelining: Overlap loads for tile N+1\n");
    printf("     with compute for tile N\n");
    printf("  4. CuTe Layout Algebra: Describes how data is laid\n");
    printf("     out in memory and mapped to threads/tiles\n\n");

    printf("Getting started:\n");
    printf("  git clone https://github.com/NVIDIA/cutlass.git\n");
    printf("  cd cutlass && mkdir build && cd build\n");
    printf("  cmake .. -DCUTLASS_NVCC_ARCHS=\"80;90;100\"\n");
    printf("  make -j\n\n");
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *      SECTION 7: CUDA TILE PROGRAMMING MODEL (CUDA 13.1+)
 * ═══════════════════════════════════════════════════════════════════
 *
 * CUDA Tile is a new programming model where developers think in terms
 * of data tiles rather than individual threads. The compiler automatically
 * maps tiles to:
 *   - Tensor cores (for matrix operations)
 *   - Shared memory (for data staging)
 *   - TMA (for bulk data movement on Hopper/Blackwell)
 *   - Optimal thread configurations
 *
 * Requirements:
 *   - CUDA Toolkit 13.3+
 *   - GPU with compute capability 8.0+ (Ampere or later)
 *   - Compile with: nvcc -std=c++20 --enable-tile
 *
 * The tile model is NOT a replacement for thread-level programming.
 * Rather, it is a higher-level interface for common patterns (GEMM,
 * reductions, scans) where the compiler can generate highly optimized
 * code. Understanding threads is still essential for debugging and
 * for workloads that don't fit the tile model.
 *
 * Conceptual CUDA Tile GEMM (pseudocode, requires cuda_tile.h):
 *
 *   #include <cuda_tile.h>
 *
 *   __tile_kernel__ void gemm_tile(float *A, float *B, float *C,
 *                                  int M, int N, int K) {
 *       // Declare tile shapes
 *       auto tileA = cuda::tile::load<128, 32>(A, M, K);  // 128x32 tile
 *       auto tileB = cuda::tile::load<32, 128>(B, K, N);  // 32x128 tile
 *
 *       // Matrix multiply-accumulate: compiler picks optimal TC ops
 *       auto tileC = cuda::tile::mma(tileA, tileB);
 *
 *       // Store result
 *       cuda::tile::store(C, tileC, M, N);
 *   }
 *
 * On Ampere, the compiler maps this to HMMA instructions with shared
 * memory staging and cp.async. On Hopper, it uses WGMMA + TMA. On
 * Blackwell, it uses tcgen05.mma + TMEM + TMA. Same source code,
 * optimal for each architecture.
 *
 * See: https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/
 */

void describeCudaTile() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 7: CUDA Tile Programming Model (CUDA 13.1+)\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("CUDA Tile is the future of GPU kernel development.\n");
    printf("It raises the abstraction level from threads to tiles.\n\n");

    printf("Traditional SIMT vs Tile programming:\n\n");

    printf("  SIMT (thread-centric):              Tile (data-centric):\n");
    printf("  ─────────────────────               ─────────────────────\n");
    printf("  int tid = ...;                      auto tile = load<M,K>(A);\n");
    printf("  __shared__ float smem[];             auto result = mma(tA, tB);\n");
    printf("  if (tid < N) {                      store(C, result);\n");
    printf("    smem[tid] = global[tid];\n");
    printf("    __syncthreads();\n");
    printf("    // manual MMA setup...\n");
    printf("  }\n\n");

    printf("Benefits of CUDA Tile:\n");
    printf("  1. Architecture-portable: same code for sm_80 through sm_100+\n");
    printf("  2. Auto-optimized: compiler handles TMA, shared mem, TC mapping\n");
    printf("  3. Less error-prone: no manual indexing, sync, or bank conflicts\n");
    printf("  4. Profileable: works with Nsight Compute same as SIMT kernels\n\n");

    printf("Limitations:\n");
    printf("  1. Only for structured patterns (GEMM, reduction, scan, etc.)\n");
    printf("  2. Less control than hand-written SIMT kernels\n");
    printf("  3. Requires CUDA 13.3+ and compute capability 8.0+\n");
    printf("  4. Thread-level knowledge still needed for debugging\n\n");

    printf("Compile a tile kernel:\n");
    printf("  nvcc -std=c++20 --enable-tile -arch=sm_120 tile_kernel.cu\n\n");
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *   SECTION 8: PERFORMANCE BEST PRACTICES FOR MODERN GPUS (2026)
 * ═══════════════════════════════════════════════════════════════════
 *
 * A summary of current best practices, incorporating lessons from
 * Ampere, Ada, Hopper, and Blackwell architectures.
 */

void demonstrateBestPractices() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Section 8: Modern CUDA Best Practices (2026)\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Your GPU: %s (%d SMs, %.1f GB, CC %d.%d)\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.major, prop.minor);

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              MODERN CUDA BEST PRACTICES (2026)               ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. MEMORY                                                   ║\n");
    printf("║     - Coalesce global memory accesses (always)               ║\n");
    printf("║     - Use async copy / TMA for shared memory staging         ║\n");
    printf("║     - Use cudaMallocAsync for dynamic allocation             ║\n");
    printf("║     - Use Unified Memory with cudaMemAdvise hints            ║\n");
    printf("║     - Prefer pinned memory for host-device transfers         ║\n");
    printf("║     - Use L2 persistence (Ampere+) for hot working sets      ║\n");
    printf("║                                                              ║\n");
    printf("║  2. COMPUTE                                                  ║\n");
    printf("║     - Use Tensor Cores for matmul (cuBLAS/CUTLASS/Tile)      ║\n");
    printf("║     - Choose the right precision (BF16 train, FP8/FP4 infer) ║\n");
    printf("║     - Accumulate in FP32 even when computing in lower prec   ║\n");
    printf("║     - Minimize warp divergence                               ║\n");
    printf("║     - Use fused multiply-add: fmaf(a, b, c) not a*b+c        ║\n");
    printf("║     - Use CompileIQ (CUDA 13.3+) for auto-tuning hot kernels ║\n");
    printf("║                                                              ║\n");
    printf("║  3. EXECUTION                                                ║\n");
    printf("║     - CUDA Graphs for repeated kernel sequences              ║\n");
    printf("║     - Multiple streams for concurrent kernels + transfers    ║\n");
    printf("║     - Grid-stride loops for flexible work distribution       ║\n");
    printf("║     - Cooperative Groups for composable synchronization      ║\n");
    printf("║     - Persistent kernels for latency-sensitive workloads     ║\n");
    printf("║                                                              ║\n");
    printf("║  4. PROFILING                                                ║\n");
    printf("║     - Nsight Systems first (find system-level bottlenecks)   ║\n");
    printf("║     - Nsight Compute second (optimize individual kernels)    ║\n");
    printf("║     - Compute Sanitizer for correctness (replaces memcheck)  ║\n");
    printf("║     - Profile on representative data sizes                   ║\n");
    printf("║     - Check the roofline model to know your bound            ║\n");
    printf("║                                                              ║\n");
    printf("║  5. PORTABILITY                                              ║\n");
    printf("║     - Use #if __CUDA_ARCH__ for compile-time feature checks  ║\n");
    printf("║     - Use cudaDeviceProp for runtime feature detection       ║\n");
    printf("║     - Target multiple architectures: -gencode arch=...       ║\n");
    printf("║     - CUDA Tile C++ for architecture-portable kernels        ║\n");
    printf("║     - Plan for Maxwell/Pascal/Volta EOL                      ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("Blackwell-specific optimization tips:\n");
    printf("  - Use MXFP4 format for inference: 2x throughput vs FP8\n");
    printf("  - Leverage CTA pair execution for shared operand access\n");
    printf("  - TMEM reduces register/shared memory pressure for MMA\n");
    printf("  - NVLink 5 at 1.8 TB/s enables efficient multi-GPU scaling\n");
    printf("  - 192 GB HBM3e at 8 TB/s: can serve 70B+ models on 1 GPU\n\n");

    printf("Hopper optimization tips:\n");
    printf("  - TMA offloads address computation from CUDA cores\n");
    printf("  - WGMMA (128 threads) for peak tensor core utilization\n");
    printf("  - Thread Block Clusters enable distributed shared memory\n");
    printf("  - Warp specialization: separate producer and consumer warps\n");
    printf("  - Software pipelining across TMA stages\n\n");
}


/*
 * ═══════════════════════════════════════════════════════════════════
 *                         MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   CUDA Tutorial: Modern CUDA Programming (2024-2026)  ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable GPU found!\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Running on: %s (Compute Capability %d.%d)\n\n",
           prop.name, prop.major, prop.minor);

    demonstrateCudaGraphs();
    printf("\n");

    demonstrateCooperativeGroups();
    printf("\n");

    demonstrateMixedPrecision();
    printf("\n");

    demonstrateStreamOrderedAlloc();
    printf("\n");

    demonstrateArchAwareness();
    printf("\n");

    describeCutlassAndCute();
    printf("\n");

    describeCudaTile();
    printf("\n");

    demonstrateBestPractices();

    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║              Tutorial 22 Complete!                    ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║                                                       ║\n");
    printf("║  Next steps to deepen your modern CUDA knowledge:     ║\n");
    printf("║                                                       ║\n");
    printf("║  1. Try CUDA Tile C++:                                ║\n");
    printf("║     docs.nvidia.com/cuda/cuda-tile-cpp-api-reference  ║\n");
    printf("║                                                       ║\n");
    printf("║  2. Explore CUTLASS examples:                         ║\n");
    printf("║     github.com/NVIDIA/cutlass/tree/main/examples      ║\n");
    printf("║                                                       ║\n");
    printf("║  3. Profile with Nsight:                              ║\n");
    printf("║     nsys profile ./22_modern_cuda                     ║\n");
    printf("║     ncu --set full ./22_modern_cuda                   ║\n");
    printf("║                                                       ║\n");
    printf("║  4. Read the Blackwell architecture whitepaper        ║\n");
    printf("║                                                       ║\n");
    printf("║  5. Try CompileIQ auto-tuning (CUDA 13.3+):           ║\n");
    printf("║     nvcc --compileiq my_kernel.cu                     ║\n");
    printf("║                                                       ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");

    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Modify the CUDA Graph example to include a cudaMemcpy node
 *    alongside the kernel nodes. Measure the improvement.
 *
 * 2. Implement a cooperative groups grid-level reduction using
 *    cudaLaunchCooperativeKernel. Compare with the block-level version.
 *
 * 3. Write a kernel that uses half2 (vectorized FP16) to process
 *    two elements per thread. Measure the speedup vs scalar FP16.
 *
 * 4. Use cudaOccupancyMaxPotentialBlockSize to automatically choose
 *    block size for each kernel, then compare with fixed 256.
 *
 * 5. If you have a Hopper or Blackwell GPU, explore the PTX
 *    output of your kernels to see which instructions are generated:
 *      nvcc -ptx -arch=sm_90 22_modern_cuda.cu
 *      grep "mma\|wgmma\|tcgen05" 22_modern_cuda.ptx
 *
 * 6. (Advanced) Install CUTLASS and run the provided GEMM examples.
 *    Compare performance with cuBLAS for various matrix sizes and
 *    precisions (FP32, FP16, FP8).
 *
 * 7. (CUDA 13.3+) Write a simple CUDA Tile kernel for vector
 *    addition or matrix transpose. Compare the generated PTX
 *    with your hand-written SIMT version.
 *
 * ═══════════════════════════════════════════════════════════════════
 */
