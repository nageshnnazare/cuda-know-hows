/*
 * CUDA Tutorial - Part 8: Advanced Topics
 * 
 * This file demonstrates:
 * 1. Atomic operations
 * 2. Warp-level primitives
 * 3. Dynamic parallelism
 * 4. Unified Memory
 * 5. Cooperative groups
 * 6. Performance optimization tips
 *
 * Compile: nvcc -o advanced 08_advanced_topics.cu -arch=sm_70 -rdc=true
 * Run:     ./advanced
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
 *                      ATOMIC OPERATIONS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Atomic operations perform read-modify-write as a single operation.
 * Essential for preventing race conditions when multiple threads
 * access the same memory location.
 *
 * Common Atomic Operations:
 * ────────────────────────
 *   atomicAdd(addr, val)    - Atomic addition
 *   atomicSub(addr, val)    - Atomic subtraction
 *   atomicMin(addr, val)    - Atomic minimum
 *   atomicMax(addr, val)    - Atomic maximum
 *   atomicExch(addr, val)   - Atomic exchange
 *   atomicCAS(addr, cmp, val) - Compare-and-swap
 *   atomicAnd/Or/Xor        - Bitwise operations
 *
 * Example: Histogram
 * ──────────────────
 * 
 * WITHOUT ATOMICS (Race condition):
 * ─────────────────────────────────
 * Thread 0: Read bin[5] = 10
 * Thread 1: Read bin[5] = 10
 * Thread 0: Write bin[5] = 11
 * Thread 1: Write bin[5] = 11  ← Lost update!
 *
 * WITH ATOMICS (Correct):
 * ───────────────────────
 * Thread 0: atomicAdd(&bin[5], 1) → 10→11
 * Thread 1: atomicAdd(&bin[5], 1) → 11→12 ✓
 */

__global__ void histogramAtomic(int *data, int *histogram, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = data[idx] % bins;
        atomicAdd(&histogram[bin], 1);
    }
}

// Global sum using atomics
__global__ void globalSumAtomic(float *input, float *result, int n) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and reduce in shared memory
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread atomically adds to global result
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    WARP-LEVEL PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Modern CUDA provides warp-level operations that are faster than
 * shared memory for intra-warp communication.
 *
 * Warp Size: 32 threads
 * ──────────────────────
 * Threads are grouped into warps of 32. All threads in a warp
 * execute the same instruction (SIMT model).
 *
 * Warp Shuffle Operations:
 * ────────────────────────
 *   __shfl_sync()      - Arbitrary shuffle
 *   __shfl_up_sync()   - Shift up
 *   __shfl_down_sync() - Shift down
 *   __shfl_xor_sync()  - XOR-based shuffle
 *
 * Warp Vote Operations:
 * ─────────────────────
 *   __all_sync()       - True if all threads have true
 *   __any_sync()       - True if any thread has true
 *   __ballot_sync()    - Collect predicate from all threads
 *
 * Example: Warp Reduction
 * ───────────────────────
 * 
 * Thread Values: [8] [4] [2] [1] [7] [3] [6] [5] ...
 *                 │   │   │   │   │   │   │   │
 * Step 1 (d=16):  └──12───┘   └───8───┘           (__shfl_down)
 *                     │           │
 * Step 2 (d=8):       └─────20────┘
 *                           │
 * Step 3 (d=4):       (...continues...)
 */

__inline__ __device__ float warpReduceSum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void warpReductionSum(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    // Load data
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction
    val = warpReduceSum(val);
    
    // Shared memory for inter-warp communication
    __shared__ float warpSums[32];
    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warpId == 0) {
        val = (lane < blockDim.x / 32) ? warpSums[lane] : 0.0f;
        val = warpReduceSum(val);
        
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      DYNAMIC PARALLELISM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Dynamic parallelism allows GPU threads to launch new kernels.
 * Useful for recursive algorithms and adaptive computations.
 *
 * Requirements:
 * ────────────
 * - Compute capability ≥ 3.5
 * - Compile with -rdc=true (relocatable device code)
 * - Link with -lcudadevrt
 *
 * Example: Recursive Quicksort
 * ────────────────────────────
 * 
 * Parent Kernel
 *     │
 *     ├── Child Kernel (left partition)
 *     │       └── Grandchild kernels...
 *     │
 *     └── Child Kernel (right partition)
 *             └── Grandchild kernels...
 */

#define DYNAMIC_THRESHOLD 32

__global__ void simpleChildKernel(int *data, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("  Child kernel at depth %d launched by block %d\n", 
               depth, blockIdx.x);
    }
}

__global__ void dynamicParallelismDemo(int *data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        printf("Parent kernel at depth %d\n", depth);
        
        // Launch child kernel from GPU
        if (depth < 3) {
            simpleChildKernel<<<2, 32>>>(data, depth + 1);
            cudaDeviceSynchronize();  // Wait for child
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                        UNIFIED MEMORY
 * ═══════════════════════════════════════════════════════════════════
 *
 * Unified Memory creates a single memory space accessible from both
 * CPU and GPU. The system automatically migrates data as needed.
 *
 * Benefits:
 * ────────
 * • Simplified programming (no explicit transfers)
 * • Automatic data migration
 * • Can exceed GPU memory size
 * • Easier development and debugging
 *
 * Drawbacks:
 * ─────────
 * • Potential performance overhead
 * • Less control over data placement
 * • Page faults on first access
 *
 * Memory Migration:
 * ────────────────
 * 
 * CPU Access:  [CPU Memory] ← Data migrates here
 *                   ↕
 * GPU Access:  [GPU Memory] ← Data migrates here
 */

__global__ void unifiedMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]) * 2.0f;
    }
}

void demonstrateUnifiedMemory() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example: Unified Memory\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    // Allocate unified memory
    float *unified_data;
    CUDA_CHECK(cudaMallocManaged(&unified_data, bytes));
    
    printf("Allocated %zu MB of unified memory\n", bytes / (1024 * 1024));
    
    // Initialize on CPU
    printf("Initializing data on CPU...\n");
    for (int i = 0; i < N; i++) {
        unified_data[i] = i * 1.0f;
    }
    
    // Process on GPU (automatic migration)
    printf("Processing on GPU...\n");
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    unifiedMemoryKernel<<<gridSize, blockSize>>>(unified_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Access on CPU (automatic migration back)
    printf("Verifying on CPU...\n");
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        float expected = sqrtf(i * 1.0f) * 2.0f;
        if (fabsf(unified_data[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    printf("Verification: %s\n\n", correct ? "✓ PASSED" : "✗ FAILED");
    
    printf("💡 Unified Memory advantages:\n");
    printf("   - No explicit cudaMemcpy needed\n");
    printf("   - Single pointer works on CPU and GPU\n");
    printf("   - Automatic migration handled by driver\n");
    printf("   - Simplifies code development\n\n");
    
    CUDA_CHECK(cudaFree(unified_data));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      COOPERATIVE GROUPS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Cooperative Groups provides flexible thread grouping beyond
 * traditional blocks and warps.
 *
 * Group Types:
 * ───────────
 * • thread_block - Traditional block
 * • thread_block_tile<Size> - Fixed-size tiles
 * • grid_group - Entire grid
 * • coalesced_group - Active threads in warp
 */

__global__ void cooperativeGroupsDemo(float *data, int n) {
    // Get the thread block group
    cg::thread_block block = cg::this_thread_block();
    
    // Get a tile of 32 threads (warp)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Warp-level reduction using cooperative groups
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            val += tile32.shfl_down(val, offset);
        }
        
        // First thread in tile writes result
        if (tile32.thread_rank() == 0) {
            data[idx / 32] = val;
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    PERFORMANCE OPTIMIZATION TIPS
 * ═══════════════════════════════════════════════════════════════════
 */

__global__ void optimizationExamples(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TIP 1: Use fast math functions
        // __sinf(), __cosf(), __expf() are faster but less accurate
        float val = input[idx];
        float result = __sinf(val) + __cosf(val);  // Fast intrinsics
        
        // TIP 2: Loop unrolling
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            result += val * i;
        }
        
        // TIP 3: Avoid division (use multiplication by reciprocal)
        float inv = __frcp_rn(3.0f);  // Fast reciprocal
        result *= inv;  // Instead of result / 3.0f
        
        // TIP 4: Avoid expensive operations
        // Use rsqrtf() instead of 1.0f/sqrtf()
        result += rsqrtf(val);
        
        output[idx] = result;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║         CUDA Tutorial: Advanced Topics               ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("📊 Device: %s\n", prop.name);
    printf("   Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("   Unified Addressing: %s\n", 
           prop.unifiedAddressing ? "✓ Yes" : "✗ No");
    printf("   Managed Memory: %s\n",
           prop.managedMemory ? "✓ Yes" : "✗ No");
    printf("   Cooperative Launch: %s\n\n",
           prop.cooperativeLaunch ? "✓ Yes" : "✗ No");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Atomic Operations
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Atomic Operations\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 20;
    const int BINS = 256;
    
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_hist = (int*)malloc(BINS * sizeof(int));
    
    // Generate random data
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % BINS;
    }
    memset(h_hist, 0, BINS * sizeof(int));
    
    int *d_data, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(int)));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    histogramAtomic<<<gridSize, blockSize>>>(d_data, d_hist, N, BINS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float atomicTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&atomicTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Verify
    int cpuSum = 0, gpuSum = 0;
    for (int i = 0; i < BINS; i++) {
        gpuSum += h_hist[i];
    }
    cpuSum = N;
    
    printf("Histogram computation:\n");
    printf("  Elements: %d\n", N);
    printf("  Bins: %d\n", BINS);
    printf("  Time: %.3f ms\n", atomicTime);
    printf("  Verification: %s (sum=%d)\n\n", 
           (cpuSum == gpuSum) ? "✓ PASSED" : "✗ FAILED", gpuSum);
    
    free(h_data);
    free(h_hist);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Warp-Level Primitives
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Warp-Level Primitives\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_result;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    
    CUDA_CHECK(cudaEventRecord(start));
    warpReductionSum<<<gridSize, blockSize>>>(d_input, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float warpTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&warpTime, start, stop));
    
    float h_result = 0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Warp-level reduction:\n");
    printf("  Elements: %d\n", N);
    printf("  Time: %.3f ms\n", warpTime);
    printf("  Result: %.0f (expected: %.0f)\n", h_result, (float)N);
    printf("  Verification: %s\n\n", 
           (fabsf(h_result - N) < 1.0f) ? "✓ PASSED" : "✗ FAILED");
    
    printf("💡 Warp primitives benefits:\n");
    printf("   - No shared memory needed\n");
    printf("   - No __syncthreads() overhead\n");
    printf("   - Faster for warp-level operations\n");
    printf("   - Implicit synchronization within warp\n\n");
    
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Dynamic Parallelism
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Dynamic Parallelism\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    if (prop.major >= 3 && prop.minor >= 5) {
        printf("Dynamic parallelism demonstration:\n\n");
        
        int *d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(int)));
        
        dynamicParallelismDemo<<<1, 1>>>(d_dummy, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        printf("\n💡 Dynamic parallelism use cases:\n");
        printf("   - Recursive algorithms (quicksort, tree traversal)\n");
        printf("   - Adaptive mesh refinement\n");
        printf("   - Dynamic work generation\n");
        printf("   - Irregular parallelism\n\n");
        
        CUDA_CHECK(cudaFree(d_dummy));
    } else {
        printf("⚠️  Dynamic parallelism requires compute capability ≥ 3.5\n");
        printf("   Current device: %d.%d\n\n", prop.major, prop.minor);
    }
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 4: Unified Memory
     * ───────────────────────────────────────────────────────────────
     */
    
    if (prop.managedMemory) {
        demonstrateUnifiedMemory();
    } else {
        printf("═══════════════════════════════════════════════════════\n");
        printf("Unified Memory: Not supported on this device\n");
        printf("═══════════════════════════════════════════════════════\n\n");
    }
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Advanced Topics Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Advanced Optimization Techniques\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("1. MEMORY OPTIMIZATIONS:\n");
    printf("   • Coalesced access patterns\n");
    printf("   • Shared memory with padding\n");
    printf("   • Texture memory for spatial locality\n");
    printf("   • Constant memory for read-only data\n");
    printf("   • Unified memory for ease of development\n\n");
    
    printf("2. COMPUTATION OPTIMIZATIONS:\n");
    printf("   • Fast math intrinsics (__sinf, __cosf)\n");
    printf("   • Warp-level primitives (shuffle, vote)\n");
    printf("   • Loop unrolling (#pragma unroll)\n");
    printf("   • Fused multiply-add (FMA)\n");
    printf("   • Avoid branch divergence\n\n");
    
    printf("3. PARALLELISM TECHNIQUES:\n");
    printf("   • Streams for concurrency\n");
    printf("   • Dynamic parallelism for recursion\n");
    printf("   • Cooperative groups for flexibility\n");
    printf("   • Multi-GPU for large problems\n\n");
    
    printf("4. PROFILING TOOLS:\n");
    printf("   • NVIDIA Nsight Systems - System-level profiling\n");
    printf("   • NVIDIA Nsight Compute - Kernel-level profiling\n");
    printf("   • nvprof - Command-line profiler\n");
    printf("   • CUDA-MEMCHECK - Memory debugger\n\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Use atomics carefully (they serialize access)     ║\n");
    printf("║ 2. Warp primitives are faster than shared memory     ║\n");
    printf("║ 3. Dynamic parallelism adds overhead               ║\n");
    printf("║ 4. Unified Memory simplifies development             ║\n");
    printf("║ 5. Always profile before optimizing                  ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement parallel merge sort using dynamic parallelism
 * 2. Create an optimized sparse matrix-vector multiplication
 * 3. Implement a lock-free queue using atomics
 * 4. Use cooperative groups for block-wide reduction
 * 5. Profile your kernels with Nsight Compute
 * 6. Implement a custom warp-level scan operation
 *
 * ═══════════════════════════════════════════════════════════════════
 *                      FURTHER READING
 * ═══════════════════════════════════════════════════════════════════
 *
 * • CUDA C++ Programming Guide:
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * • CUDA Best Practices Guide:
 *   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
 *
 * • Cooperative Groups:
 *   https://developer.nvidia.com/blog/cooperative-groups/
 *
 * • GPU Performance Analysis:
 *   https://developer.nvidia.com/blog/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
 *
 * ═══════════════════════════════════════════════════════════════════
 */

