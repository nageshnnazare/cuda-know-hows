/*
 * CUDA Tutorial - Part 3: CUDA Memory Model
 * 
 * This file demonstrates:
 * 1. Different types of GPU memory
 * 2. Memory coalescing and alignment
 * 3. Global, constant, and texture memory
 * 4. Performance implications of memory access patterns
 * 5. Practical examples with performance comparison
 *
 * Compile: nvcc -o memory_model 03_memory_model.cu
 * Run:     ./memory_model
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
 *                      MEMORY HIERARCHY OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Memory Type    | Scope    | Lifetime    | Access Speed | Size
 * ─────────────────────────────────────────────────────────────────
 * Register       | Thread   | Thread      | 1 cycle      | ~64KB/SM
 * Local Memory   | Thread   | Thread      | ~400 cycles  | Limited
 * Shared Memory  | Block    | Block       | ~5 cycles    | 48-96KB/SM
 * Global Memory  | Grid     | Application | ~400 cycles  | GBs
 * Constant Memory| Grid     | Application | ~5 cycles*   | 64KB
 * Texture Memory | Grid     | Application | ~5 cycles*   | Device
 * 
 * *when cached
 *
 * Memory Access Visualization:
 * 
 *     Fast ↑                         Small ↑
 *          │                               │
 *    [Registers]──────────────────────[Registers]
 *          │                               │
 *    [Shared Memory]────────────────[Shared Memory]
 *          │                               │
 *    [L1/L2 Cache]──────────────────[L1/L2 Cache]
 *          │                               │
 *    [Global Memory]────────────────[Global Memory]
 *          │                               │
 *    Slow  ↓                         Large ↓
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    CONSTANT MEMORY DEMONSTRATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Constant memory is:
 * - Read-only from device code
 * - Cached for fast access when all threads read same location
 * - Limited to 64KB
 * - Best for uniform access patterns
 */

// Constant memory declaration (visible to all kernels)
__constant__ float const_array[256];

// Compare global vs constant memory access
__global__ void globalMemoryAccess(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int idx = tid % 256;
        output[tid] = input[tid] * input[idx];  // Global memory read
    }
}

__global__ void constantMemoryAccess(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int idx = tid % 256;
        output[tid] = input[tid] * const_array[idx];  // Constant memory read
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   MEMORY COALESCING DEMONSTRATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Memory coalescing occurs when threads in a warp access consecutive
 * memory locations. This allows the GPU to combine multiple memory
 * requests into fewer transactions.
 *
 * COALESCED ACCESS (Good):
 * ┌────────────────────────────────────────────────────┐
 * │ Warp (32 threads)                                  │
 * ├────────────────────────────────────────────────────┤
 * │ T0  T1  T2  T3  ... T29 T30 T31                    │
 * │  ↓   ↓   ↓   ↓       ↓   ↓   ↓                     │
 * │ [0] [1] [2] [3] ... [29][30][31]  Memory           │
 * │  └───────────────────────────────→ 1 Transaction   │
 * └────────────────────────────────────────────────────┘
 *
 * STRIDED ACCESS (Bad):
 * ┌────────────────────────────────────────────────────┐
 * │ Warp (32 threads)                                  │
 * ├────────────────────────────────────────────────────┤
 * │ T0  T1  T2  T3  ... T29 T30 T31                    │
 * │  ↓   ↓   ↓   ↓       ↓   ↓   ↓                     │
 * │ [0] [2] [4] [6] ... [58][60][62]  Memory           │
 * │  └→  └→  └→  └→      └→  └→  └→ 32 Transactions    │
 * └────────────────────────────────────────────────────┘
 */

// Good: Coalesced access
__global__ void coalescedAccess(float *data, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Consecutive threads access consecutive memory locations
        result[tid] = data[tid] * 2.0f;
    }
}

// Bad: Strided access
__global__ void stridedAccess(float *data, float *result, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    
    if (idx < n) {
        // Threads access memory with large gaps (poor coalescing)
        result[idx] = data[idx] * 2.0f;
    }
}

// Worst: Random access
__global__ void randomAccess(float *data, float *result, int *indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Completely random access pattern
        int idx = indices[tid];
        if (idx < n) {
            result[tid] = data[idx] * 2.0f;
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      MEMORY ALIGNMENT EXAMPLE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Aligned memory accesses are more efficient. CUDA prefers 128-byte
 * aligned memory for optimal performance.
 */

__global__ void alignedVsUnaligned(float *aligned, float *unaligned, 
                                   float *result, int n, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Aligned access
        result[tid] = aligned[tid];
        
        // Unaligned access (offset breaks alignment)
        if (tid + offset < n) {
            result[tid] += unaligned[tid + offset];
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    BANK CONFLICTS (Preview)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Shared memory is divided into banks. Multiple threads accessing the
 * same bank (but different addresses) causes bank conflicts.
 *
 * NO CONFLICT:
 * ┌──────────────────────────────────────────┐
 * │ Threads: T0  T1  T2  T3  T4  T5  T6  T7  │
 * │           ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓  │
 * │ Banks:   B0  B1  B2  B3  B4  B5  B6  B7  │
 * │ Access: [0] [1] [2] [3] [4] [5] [6] [7]  │
 * └──────────────────────────────────────────┘
 * Result: All accesses in parallel (fast)
 *
 * 2-WAY BANK CONFLICT:
 * ┌──────────────────────────────────────────┐
 * │ Threads: T0  T1  T2  T3  T4  T5  T6  T7  │
 * │           ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓  │
 * │ Banks:   B0  B0  B1  B1  B2  B2  B3  B3  │
 * │ Access: [0] [8] [1] [9] [2][10] [3][11]  │
 * └──────────────────────────────────────────┘
 * Result: 2 serialized accesses (2x slower)
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                        TIMING UTILITIES
 * ═══════════════════════════════════════════════════════════════════
 */

float timeKernel(void (*kernel)(float*, float*, int), 
                 float *d_input, float *d_output, int n,
                 int blockSize, const char *name) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Warm-up
    kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds / 100.0f;  // Average time
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║         CUDA Tutorial: Memory Model Deep Dive         ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Array size
    const int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.2f MB)\n\n", N, bytes / (1024.0 * 1024.0));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Constant Memory vs Global Memory
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Constant Memory vs Global Memory\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Allocate memory
    float *h_input = (float*)malloc(bytes);
    float *h_coeff = (float*)malloc(256 * sizeof(float));
    float *h_output = (float*)malloc(bytes);
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / N;
    }
    for (int i = 0; i < 256; i++) {
        h_coeff[i] = sin((float)i);
    }
    
    // Device memory
    float *d_input, *d_coeff, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy data
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeff, h_coeff, 256 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(const_array, h_coeff, 256 * sizeof(float)));
    
    // Test global memory
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    globalMemoryAccess<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float globalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&globalTime, start, stop));
    
    // Test constant memory
    CUDA_CHECK(cudaEventRecord(start));
    constantMemoryAccess<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float constantTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&constantTime, start, stop));
    
    printf("Global memory time:   %.3f ms\n", globalTime);
    printf("Constant memory time: %.3f ms\n", constantTime);
    printf("Speedup: %.2fx\n\n", globalTime / constantTime);
    
    printf("💡 Explanation:\n");
    printf("   Constant memory is cached and optimized for broadcast\n");
    printf("   (all threads reading same value). This test shows the\n");
    printf("   benefit when threads frequently read the same data.\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Memory Coalescing
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Memory Coalescing Impact\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Test coalesced access
    CUDA_CHECK(cudaEventRecord(start));
    coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float coalescedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&coalescedTime, start, stop));
    
    // Test strided access (stride = 2)
    CUDA_CHECK(cudaEventRecord(start));
    stridedAccess<<<gridSize, blockSize>>>(d_input, d_output, N, 2);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float stridedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&stridedTime, start, stop));
    
    // Test random access
    int *h_indices = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_indices[i] = (i * 7919) % N;  // Pseudo-random
    }
    
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    randomAccess<<<gridSize, blockSize>>>(d_input, d_output, d_indices, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float randomTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&randomTime, start, stop));
    
    printf("┌────────────────────────┬──────────────┬───────────┐\n");
    printf("│ Access Pattern         │ Time (ms)    │ Speedup   │\n");
    printf("├────────────────────────┼──────────────┼───────────┤\n");
    printf("│ Coalesced (Sequential) │ %8.3f     │ 1.00x     │\n", coalescedTime);
    printf("│ Strided (Stride=2)     │ %8.3f     │ %.2fx     │\n", 
           stridedTime, coalescedTime / stridedTime);
    printf("│ Random                 │ %8.3f     │ %.2fx     │\n", 
           randomTime, coalescedTime / randomTime);
    printf("└────────────────────────┴──────────────┴───────────┘\n\n");
    
    printf("💡 Explanation:\n");
    printf("   Coalesced access allows GPU to combine multiple memory\n");
    printf("   requests into single transactions. Strided and random\n");
    printf("   access patterns prevent this optimization, resulting\n");
    printf("   in significantly slower performance.\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Memory Bandwidth Calculation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Memory Bandwidth Analysis\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    float dataGB = (2.0f * bytes) / (1024.0f * 1024.0f * 1024.0f);  // Read + Write
    float bandwidthCoalesced = dataGB / (coalescedTime / 1000.0f);
    float bandwidthStrided = dataGB / (stridedTime / 1000.0f);
    float bandwidthRandom = dataGB / (randomTime / 1000.0f);
    
    // Get theoretical peak bandwidth
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peakBandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    
    printf("GPU: %s\n", prop.name);
    printf("Theoretical Peak Bandwidth: %.1f GB/s\n\n", peakBandwidth);
    
    printf("┌────────────────────────┬──────────────┬─────────────┐\n");
    printf("│ Access Pattern         │ Bandwidth    │ Efficiency  │\n");
    printf("├────────────────────────┼──────────────┼─────────────┤\n");
    printf("│ Coalesced              │ %7.1f GB/s │ %6.1f%%    │\n", 
           bandwidthCoalesced, 100.0f * bandwidthCoalesced / peakBandwidth);
    printf("│ Strided                │ %7.1f GB/s │ %6.1f%%    │\n", 
           bandwidthStrided, 100.0f * bandwidthStrided / peakBandwidth);
    printf("│ Random                 │ %7.1f GB/s │ %6.1f%%    │\n", 
           bandwidthRandom, 100.0f * bandwidthRandom / peakBandwidth);
    printf("└────────────────────────┴──────────────┴─────────────┘\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Display Visual Memory Hierarchy
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("GPU Memory Hierarchy Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("  ┌─────────────────────────────────────────────┐\n");
    printf("  │         CUDA MEMORY HIERARCHY               │\n");
    printf("  ├─────────────────────────────────────────────┤\n");
    printf("  │                                             │\n");
    printf("  │  Speed    │ Memory Type │ Size      │ Scope│\n");
    printf("  │  ═══════════════════════════════════════════│\n");
    printf("  │  ⚡⚡⚡     │ Registers   │ ~64KB/SM  │ Thrd │\n");
    printf("  │  ⚡⚡⚡     │ Shared Mem  │ ~48KB/SM  │ Blck │\n");
    printf("  │  ⚡⚡      │ L1 Cache    │ ~128KB/SM │ Auto │\n");
    printf("  │  ⚡⚡      │ Constant    │ 64KB      │ Grid │\n");
    printf("  │  ⚡       │ L2 Cache    │ ~6MB      │ Auto │\n");
    printf("  │  ⚡       │ Global Mem  │ %5.1f GB  │ Grid │\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  │                                             │\n");
    printf("  └─────────────────────────────────────────────┘\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_indices));
    
    free(h_input);
    free(h_coeff);
    free(h_output);
    free(h_indices);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Use constant memory for read-only data            ║\n");
    printf("║ 2. Always aim for coalesced memory access            ║\n");
    printf("║ 3. Avoid strided and random access patterns          ║\n");
    printf("║ 4. Memory bandwidth is often the bottleneck          ║\n");
    printf("║ 5. Shared memory is much faster than global memory   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement a kernel using texture memory
 * 2. Compare different stride values (2, 4, 8, 16, 32)
 * 3. Measure the impact of alignment on performance
 * 4. Implement a kernel with deliberately bad memory access
 * 5. Create a visualization of memory coalescing in your data
 *
 * ═══════════════════════════════════════════════════════════════════
 */

