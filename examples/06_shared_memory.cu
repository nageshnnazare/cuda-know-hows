/*
 * CUDA Tutorial - Part 6: Shared Memory and Optimization
 * 
 * This file demonstrates:
 * 1. Using shared memory effectively
 * 2. Avoiding bank conflicts
 * 3. Reduction algorithms
 * 4. Prefix sum (scan) operations
 * 5. Advanced optimization techniques
 *
 * Compile: nvcc -o shared_mem 06_shared_memory.cu
 * Run:     ./shared_mem
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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
 *                    SHARED MEMORY OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Shared Memory Characteristics:
 * ─────────────────────────────
 * - On-chip memory (very fast, ~100x faster than global memory)
 * - Shared among threads in the same block
 * - Limited size: 48-96KB per SM
 * - Organized into 32 banks
 * - Lifetime: kernel execution of a block
 *
 * Memory Hierarchy Speed Comparison:
 * ──────────────────────────────────
 *   Registers:      ~1 cycle
 *   Shared Memory:  ~5 cycles    ← We focus here!
 *   L1 Cache:       ~30 cycles
 *   L2 Cache:       ~200 cycles
 *   Global Memory:  ~400 cycles
 *
 * Bank Organization (32 banks on modern GPUs):
 * ────────────────────────────────────────────
 *
 *   Address    Bank
 *   ───────────────
 *   0          0
 *   1          1
 *   2          2
 *   ...        ...
 *   31         31
 *   32         0    ← wraps around
 *   33         1
 *
 * Bank Conflict Examples:
 * ───────────────────────
 *
 * NO CONFLICT (Good):
 *   Thread 0 → Bank 0
 *   Thread 1 → Bank 1
 *   Thread 2 → Bank 2
 *   All threads access different banks → 1 transaction
 *
 * 2-WAY CONFLICT (Bad):
 *   Thread 0 → Bank 0
 *   Thread 1 → Bank 0  ← Conflict!
 *   Thread 2 → Bank 1
 *   ...
 *   Result: 2 serialized transactions (2x slower)
 *
 * BROADCAST (OK):
 *   All threads read SAME address → No conflict
 */

#define BLOCK_SIZE 256
#define WARP_SIZE 32

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 1: PARALLEL REDUCTION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Problem: Sum all elements in an array
 * 
 * Sequential: sum = a[0] + a[1] + a[2] + ... + a[n-1]
 * 
 * Parallel Tree Reduction:
 * ────────────────────────
 * 
 * Input:  [1] [2] [3] [4] [5] [6] [7] [8]
 *          │   │   │   │   │   │   │   │
 * Step 1:  └─3─┘   └─7─┘   └─11┘   └─15┘
 *              │       │       │       │
 * Step 2:      └──10───┘       └──26───┘
 *                  │               │
 * Step 3:          └──────36───────┘
 *
 * Visual Process:
 * ───────────────
 * 
 * Iteration 0 (stride=1):
 *   Threads: T0  T1  T2  T3  T4  T5  T6  T7
 *   Data:    [1] [2] [3] [4] [5] [6] [7] [8]
 *   Active:  T0      T2      T4      T6
 *   Result:  [3]     [7]     [11]    [15]
 * 
 * Iteration 1 (stride=2):
 *   Active:  T0              T4
 *   Result:  [10]            [26]
 * 
 * Iteration 2 (stride=4):
 *   Active:  T0
 *   Result:  [36]
 */

// Naive reduction (has divergence issues)
__global__ void reduceNaive(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // Branch divergence!
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction (no divergence)
__global__ void reduceOptimized(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and perform first level of reduction during load
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction with no divergence
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {  // Contiguous threads active
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Highly optimized reduction with unrolling
__global__ void reduceUnrolled(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load two elements per thread and do first add
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction with loop unrolling
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float *vdata = sdata;
        if (blockDim.x >= 64) vdata[tid] += vdata[tid + 32];
        if (blockDim.x >= 32) vdata[tid] += vdata[tid + 16];
        if (blockDim.x >= 16) vdata[tid] += vdata[tid + 8];
        if (blockDim.x >= 8)  vdata[tid] += vdata[tid + 4];
        if (blockDim.x >= 4)  vdata[tid] += vdata[tid + 2];
        if (blockDim.x >= 2)  vdata[tid] += vdata[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 2: BANK CONFLICTS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Demonstrating bank conflicts with different access patterns
 */

// Sequential access - causes bank conflicts
__global__ void bankConflictBad(float *output) {
    __shared__ float shared[32][32];
    
    int tid = threadIdx.x;
    
    // Each thread accesses same column (same bank!)
    // Thread 0 → shared[0][0] (bank 0)
    // Thread 1 → shared[1][0] (bank 0)  ← Conflict!
    // Thread 2 → shared[2][0] (bank 0)  ← Conflict!
    // Results in 32-way bank conflict
    for (int i = 0; i < 32; i++) {
        shared[tid][i] = tid * i;
    }
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += shared[i][tid];  // Column access - conflicts!
    }
    
    output[tid] = sum;
}

// Padding to avoid bank conflicts
__global__ void bankConflictGood(float *output) {
    __shared__ float shared[32][32 + 1];  // +1 padding avoids conflicts
    
    int tid = threadIdx.x;
    
    // With padding, consecutive threads access different banks
    for (int i = 0; i < 32; i++) {
        shared[tid][i] = tid * i;
    }
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += shared[i][tid];  // No conflicts with padding!
    }
    
    output[tid] = sum;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 3: 1D STENCIL
 * ═══════════════════════════════════════════════════════════════════
 *
 * Stencil operations are common in scientific computing
 *
 * Example: 3-point stencil
 * ────────────────────────
 *   output[i] = input[i-1] + input[i] + input[i+1]
 *
 * Visualization:
 * ──────────────
 *   Input:  ... [a] [b] [c] [d] [e] ...
 *                ↓   ↓   ↓
 *   Output:     [a+b+c] [b+c+d] [c+d+e]
 *
 * Using shared memory to cache input data reduces global memory reads
 * from 3N to N+2 (plus 2 for boundaries)
 */

#define RADIUS 3
#define STENCIL_BLOCK 256

__global__ void stencil1D(float *input, float *output, int n) {
    __shared__ float shared[STENCIL_BLOCK + 2 * RADIUS];
    
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;
    
    // Load main data
    if (gindex < n) {
        shared[lindex] = input[gindex];
    }
    
    // Load left halo
    if (threadIdx.x < RADIUS) {
        shared[lindex - RADIUS] = (gindex >= RADIUS) ? 
                                   input[gindex - RADIUS] : 0.0f;
    }
    
    // Load right halo
    if (threadIdx.x < RADIUS) {
        shared[lindex + blockDim.x] = (gindex + blockDim.x < n) ?
                                       input[gindex + blockDim.x] : 0.0f;
    }
    
    __syncthreads();
    
    // Apply stencil
    if (gindex < n) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += shared[lindex + offset];
        }
        output[gindex] = result;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 4: PREFIX SUM (SCAN)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Prefix Sum: Given [a0, a1, a2, a3], compute:
 *   Exclusive scan: [0, a0, a0+a1, a0+a1+a2]
 *   Inclusive scan: [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3]
 *
 * Example:
 * ────────
 *   Input:     [3, 1, 7, 0, 4, 1, 6, 3]
 *   Exclusive: [0, 3, 4, 11, 11, 15, 16, 22]
 *   Inclusive: [3, 4, 11, 11, 15, 16, 22, 25]
 *
 * Hillis-Steele Algorithm (Work Inefficient but Simple):
 * ───────────────────────────────────────────────────────
 *
 * Input: [3] [1] [7] [0] [4] [1] [6] [3]
 *         │   │   │   │   │   │   │   │
 * Step1:  └─4─┘ └─8─┘ └─4─┘ └─5─┘ └─9─┘
 *         │     │     │     │     │     │
 * Step2:  └────11────┘ └────12───┘ └───15
 *         │           │           │
 * Step3:  └──────────15──────────┘
 */

__global__ void scanHillisSteele(float *input, float *output, int n) {
    __shared__ float temp[2 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    
    // Load input into shared memory
    temp[tid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Scan
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        
        if (tid >= offset) {
            temp[pout * blockDim.x + tid] = 
                temp[pin * blockDim.x + tid] + 
                temp[pin * blockDim.x + tid - offset];
        } else {
            temp[pout * blockDim.x + tid] = 
                temp[pin * blockDim.x + tid];
        }
        __syncthreads();
    }
    
    if (tid < n) {
        output[tid] = temp[pout * blockDim.x + tid];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 5: HISTOGRAM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Computing histogram using shared memory and atomic operations
 */

#define NUM_BINS 256

__global__ void histogramShared(unsigned char *input, unsigned int *output, int n) {
    __shared__ unsigned int shared_hist[NUM_BINS];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < NUM_BINS) {
        shared_hist[tid] = 0;
    }
    __syncthreads();
    
    // Compute histogram in shared memory
    if (gid < n) {
        atomicAdd(&shared_hist[input[gid]], 1);
    }
    __syncthreads();
    
    // Write to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&output[tid], shared_hist[tid]);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 */

float measureKernelTime(void (*kernel)(float*, float*, int),
                        float *d_in, float *d_out, int n,
                        int blocks, int threads) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up
    kernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        kernel<<<blocks, threads>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / 100.0f;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║      CUDA Tutorial: Shared Memory & Optimization     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Get shared memory info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("📊 Device: %s\n", prop.name);
    printf("   Shared memory per block: %zu KB\n", 
           prop.sharedMemPerBlock / 1024);
    printf("   Shared memory per SM: %zu KB\n", 
           prop.sharedMemPerMultiprocessor / 1024);
    printf("   Number of banks: 32\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Reduction Performance Comparison
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Reduction Performance\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int n = 1 << 24;  // 16M elements
    size_t bytes = n * sizeof(float);
    
    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Sum should be n
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, (n / BLOCK_SIZE + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Array size: %d elements (%.2f MB)\n", n, bytes / (1024.0f * 1024.0f));
    printf("Blocks: %d, Threads per block: %d\n\n", blocks, BLOCK_SIZE);
    
    // Test naive reduction
    float naiveTime = measureKernelTime(reduceNaive, d_input, d_output, n,
                                        blocks, BLOCK_SIZE);
    
    // Test optimized reduction
    float optTime = measureKernelTime(reduceOptimized, d_input, d_output, n,
                                      blocks, BLOCK_SIZE);
    
    // Test unrolled reduction
    float unrollTime = measureKernelTime(reduceUnrolled, d_input, d_output, n,
                                         blocks / 2, BLOCK_SIZE);
    
    printf("┌──────────────────┬────────────┬──────────────┐\n");
    printf("│ Implementation   │ Time (ms)  │ Speedup      │\n");
    printf("├──────────────────┼────────────┼──────────────┤\n");
    printf("│ Naive            │ %9.3f  │ 1.00x        │\n", naiveTime);
    printf("│ Optimized        │ %9.3f  │ %.2fx        │\n", 
           optTime, naiveTime / optTime);
    printf("│ Unrolled         │ %9.3f  │ %.2fx        │\n", 
           unrollTime, naiveTime / unrollTime);
    printf("└──────────────────┴────────────┴──────────────┘\n\n");
    
    printf("💡 Optimizations applied:\n");
    printf("   1. Eliminated thread divergence\n");
    printf("   2. Sequential addressing for coalesced access\n");
    printf("   3. Loop unrolling for reduced overhead\n");
    printf("   4. Warp-level operations without synchronization\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Bank Conflicts
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Bank Conflict Impact\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    float *d_bankTest;
    CUDA_CHECK(cudaMalloc(&d_bankTest, 32 * sizeof(float)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test with conflicts
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10000; i++) {
        bankConflictBad<<<1, 32>>>(d_bankTest);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float conflictTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&conflictTime, start, stop));
    
    // Test without conflicts
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10000; i++) {
        bankConflictGood<<<1, 32>>>(d_bankTest);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float noConflictTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&noConflictTime, start, stop));
    
    printf("Shared memory access patterns:\n\n");
    printf("┌──────────────────┬────────────┬──────────────┐\n");
    printf("│ Pattern          │ Time (ms)  │ Speedup      │\n");
    printf("├──────────────────┼────────────┼──────────────┤\n");
    printf("│ With conflicts   │ %9.3f  │ 1.00x        │\n", conflictTime);
    printf("│ With padding     │ %9.3f  │ %.2fx        │\n", 
           noConflictTime, conflictTime / noConflictTime);
    printf("└──────────────────┴────────────┴──────────────┘\n\n");
    
    printf("💡 Bank conflict avoidance:\n");
    printf("   - Padding arrays: float shared[N][M+1]\n");
    printf("   - Changes memory layout to avoid conflicts\n");
    printf("   - Small memory overhead for significant speedup\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Stencil Operation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: 1D Stencil with Shared Memory\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int stencilN = 1 << 20;
    size_t stencilBytes = stencilN * sizeof(float);
    
    float *d_stencilIn, *d_stencilOut;
    CUDA_CHECK(cudaMalloc(&d_stencilIn, stencilBytes));
    CUDA_CHECK(cudaMalloc(&d_stencilOut, stencilBytes));
    
    int stencilBlocks = (stencilN + STENCIL_BLOCK - 1) / STENCIL_BLOCK;
    
    CUDA_CHECK(cudaEventRecord(start));
    stencil1D<<<stencilBlocks, STENCIL_BLOCK>>>(d_stencilIn, d_stencilOut, stencilN);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float stencilTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&stencilTime, start, stop));
    
    float bandwidth = (2.0f * stencilBytes) / (1024.0f * 1024.0f * 1024.0f) /
                      (stencilTime / 1000.0f);
    
    printf("Stencil radius: %d\n", RADIUS);
    printf("Array size: %d elements\n", stencilN);
    printf("Processing time: %.3f ms\n", stencilTime);
    printf("Effective bandwidth: %.2f GB/s\n\n", bandwidth);
    
    printf("💡 Shared memory benefits:\n");
    printf("   - Reduces global memory reads from %dx to 1x\n", 2 * RADIUS + 1);
    printf("   - Efficient halo exchange at block boundaries\n");
    printf("   - Common pattern in image processing & PDEs\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Visualization Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Shared Memory Best Practices\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("✓ DO:\n");
    printf("  • Use shared memory to reduce global memory accesses\n");
    printf("  • Pad arrays to avoid bank conflicts\n");
    printf("  • Reuse data loaded by multiple threads\n");
    printf("  • Synchronize properly with __syncthreads()\n");
    printf("  • Consider shared memory limits when sizing blocks\n\n");
    
    printf("✗ DON'T:\n");
    printf("  • Forget __syncthreads() after writing to shared memory\n");
    printf("  • Create bank conflicts with strided access\n");
    printf("  • Exceed shared memory limits\n");
    printf("  • Use shared memory for data accessed only once\n");
    printf("  • Synchronize within divergent code paths\n\n");
    
    printf("MEMORY ACCESS PATTERNS:\n");
    printf("───────────────────────────────────────────────────────\n");
    printf("  Pattern              Bank Conflicts    Performance\n");
    printf("  ─────────────────────────────────────────────────────\n");
    printf("  shared[tid]          None              ⚡⚡⚡ Excellent\n");
    printf("  shared[tid*2]        2-way             ⚡⚡  Good\n");
    printf("  shared[tid*4]        4-way             ⚡   Fair\n");
    printf("  shared[0][tid]       32-way            ⚫   Poor\n");
    printf("  shared[tid/32][tid]  None (broadcast)  ⚡⚡⚡ Excellent\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bankTest));
    CUDA_CHECK(cudaFree(d_stencilIn));
    CUDA_CHECK(cudaFree(d_stencilOut));
    
    free(h_input);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Shared memory is ~100x faster than global memory  ║\n");
    printf("║ 2. Avoid bank conflicts with proper padding          ║\n");
    printf("║ 3. Always synchronize after writing to shared memory ║\n");
    printf("║ 4. Use for data reuse patterns (stencils, tiles)     ║\n");
    printf("║ 5. Optimize reductions with unrolling & warp ops     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement 2D stencil operation for image blurring
 * 2. Create an efficient parallel prefix sum for large arrays
 * 3. Optimize matrix transpose with shared memory and padding
 * 4. Implement min/max reduction using shared memory
 * 5. Create a 2D histogram for color images
 * 6. Measure actual bank conflicts using profiling tools
 *
 * ═══════════════════════════════════════════════════════════════════
 */

