/*
 * CUDA Tutorial - Part 17: Advanced Memory Techniques
 * 
 * This file demonstrates advanced memory management features:
 * 1. Texture Memory - Cached, filtered reads for spatial data
 * 2. Zero-Copy Memory - Direct host memory access
 * 3. Unified Memory - Automatic migration between host/device
 * 4. Pinned Memory - Fast host-device transfers
 * 5. Mapped Memory - Host-visible device memory
 * 6. Memory Pools - Efficient allocation
 *
 * Each technique includes:
 * - When to use it
 * - Performance characteristics
 * - Code examples
 * - Comparison with alternatives
 *
 * Compile: nvcc -o advanced_mem 17_advanced_memory.cu -O3
 * Run:     ./advanced_mem
 */

#include <stdio.h>
#include <stdlib.h>
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
 *                   1. TEXTURE MEMORY
 * ═══════════════════════════════════════════════════════════════════
 *
 * Texture memory provides:
 * - Automatic caching (good for spatial locality)
 * - Hardware filtering (linear interpolation)
 * - Automatic boundary handling (clamp, wrap)
 *
 * Use Cases:
 * ─────────
 * - Image processing
 * - Volume rendering
 * - Irregular access patterns with spatial locality
 *
 * Visual Comparison:
 * ─────────────────
 * Global Memory:         Texture Memory:
 * ┌──────────────┐       ┌──────────────┐
 * │  No cache    │       │  ✓ Cached    │
 * │  No filter   │       │  ✓ Filtered  │
 * │  Manual wrap │  vs   │  ✓ Auto wrap │
 * │  Raw access  │       │  ✓ Normalized│
 * └──────────────┘       └──────────────┘
 *
 * Texture Object API (Modern):
 * ───────────────────────────
 * 1. Allocate device memory
 * 2. Create texture object
 * 3. Bind to kernel parameter
 * 4. Read using tex2D(), tex3D()
 */

// Texture object (passed as kernel parameter)
__global__ void textureLookup2D(cudaTextureObject_t tex,
                                float *output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Normalized coordinates [0, 1]
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        // Hardware-filtered read (bilinear interpolation)
        float value = tex2D<float>(tex, u, v);
        
        output[y * width + x] = value;
    }
}

// Example: Bilinear interpolation for image resizing
__global__ void imageResize(cudaTextureObject_t inputTex,
                            float *output,
                            int in_width, int in_height,
                            int out_width, int out_height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x < out_width && out_y < out_height) {
        // Map output coordinates to input coordinates
        float u = (out_x + 0.5f) / out_width;
        float v = (out_y + 0.5f) / out_height;
        
        // Texture hardware does bilinear interpolation automatically!
        float value = tex2D<float>(inputTex, u, v);
        
        output[out_y * out_width + out_x] = value;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   2. ZERO-COPY MEMORY
 * ═══════════════════════════════════════════════════════════════════
 *
 * Zero-copy allows GPU to directly access host memory over PCIe.
 *
 * Memory Flow:
 * ───────────
 * Normal:                    Zero-Copy:
 * ┌──────┐                   ┌──────┐
 * │ Host │  ─copy→  ┌──────┐ │ Host │ ←─direct─┐
 * │Memory│          │Device│ │Memory│          │
 * └──────┘          │Memory│ └──────┘         GPU
 *                   └──────┘
 *                      ↑
 *                     GPU
 *
 * Advantages:
 * ──────────
 * ✓ No explicit copy needed
 * ✓ Good for data used once
 * ✓ Saves device memory
 *
 * Disadvantages:
 * ─────────────
 * ✗ Slower than device memory (PCIe bandwidth)
 * ✗ High latency
 * ✗ Not cached on GPU
 *
 * When to Use:
 * ───────────
 * - Large datasets, one-time access
 * - Insufficient device memory
 * - Streaming data
 */

__global__ void zeroCopyKernel(float *host_data, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct read from host memory (over PCIe)
        float value = host_data[idx];
        output[idx] = value * 2.0f;
    }
}

void demonstrateZeroCopy(int n) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Zero-Copy Memory Example\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory with cudaHostAlloc (pinned + mapped)
    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, bytes, 
                            cudaHostAllocMapped | cudaHostAllocWriteCombined));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    // Get device pointer to host memory
    float *d_host_ptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_host_ptr, h_data, 0));
    
    // Allocate output
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Launch kernel - reads directly from host memory!
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    zeroCopyKernel<<<gridSize, blockSize>>>(d_host_ptr, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("✓ Zero-copy kernel executed\n");
    printf("  GPU accessed host memory directly over PCIe\n");
    printf("  No cudaMemcpy needed!\n\n");
    
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_data));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   3. UNIFIED MEMORY (Managed Memory)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Unified Memory provides a single memory space accessible by both
 * CPU and GPU with automatic migration.
 *
 * Memory Management:
 * ─────────────────
 * Traditional:              Unified Memory:
 * 
 * cudaMalloc(&d_ptr, ...)   cudaMallocManaged(&ptr, ...)
 * cudaMemcpy(d_ptr, h, ...) // No copy needed!
 * kernel<<<...>>>(d_ptr)    kernel<<<...>>>(ptr)
 * cudaMemcpy(h, d_ptr, ...) // No copy needed!
 * use h_data                use ptr on host
 *
 * Visual:
 * ──────
 *   CPU Code          Unified Memory         GPU Code
 *   ────────          ──────────────         ────────
 *     ptr   ←──────── Automatic Migration ────────→   ptr
 *   (reads)                                         (writes)
 *
 * Page Migration:
 * ──────────────
 * Data automatically migrates on demand:
 * 
 * Time  CPU Access  GPU Access  Location
 * ────  ──────────  ──────────  ────────
 *  0    Write       -           Host
 *  1    -           Read        GPU (migrated)
 *  2    -           Write       GPU
 *  3    Read        -           Host (migrated)
 *
 * Prefetching:
 * ───────────
 * Hint to runtime where data will be used:
 *   cudaMemPrefetchAsync(ptr, size, deviceId)
 */

__global__ void unifiedMemKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * data[idx];  // Square each element
    }
}

void demonstrateUnifiedMemory(int n) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Unified Memory Example\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate unified memory
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    // Initialize on CPU
    printf("1. CPU initializes data...\n");
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
    
    // Optional: Prefetch to GPU for better performance
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, device));
    
    // Process on GPU
    printf("2. GPU processes data (automatic migration)...\n");
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    unifiedMemKernel<<<gridSize, blockSize>>>(data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Optional: Prefetch back to CPU
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    
    // Read on CPU (automatic migration back to host)
    printf("3. CPU reads results (automatic migration)...\n");
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {  // Just first 10
        sum += data[i];
    }
    
    printf("\n✓ Unified Memory demonstration complete\n");
    printf("  Single pointer used by both CPU and GPU\n");
    printf("  Automatic migration handled by CUDA runtime\n");
    printf("  Sum of first 10 elements: %.2f\n\n", sum);
    
    CUDA_CHECK(cudaFree(data));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   4. PINNED MEMORY
 * ═══════════════════════════════════════════════════════════════════
 *
 * Pinned (page-locked) memory cannot be paged to disk, enabling:
 * - Faster host-device transfers
 * - Concurrent copy and kernel execution
 * - Zero-copy access
 *
 * Performance:
 * ───────────
 * Pageable Memory:         Pinned Memory:
 * ┌────────────┐          ┌────────────┐
 * │  ~3 GB/s   │          │  ~12 GB/s  │  (PCIe Gen3 x16)
 * │  Staging   │    vs    │  Direct    │
 * │  required  │          │  DMA       │
 * └────────────┘          └────────────┘
 *
 * Transfer Flow:
 * ─────────────
 * Pageable:                   Pinned:
 * Host      Staging   GPU     Host         GPU
 * Memory → Buffer  → Memory   Memory  →   Memory
 *   ↓        ↓        ↓         ↓           ↓
 * copy    copy DMA             Direct DMA
 *
 * Usage:
 * ─────
 * cudaHostAlloc(&ptr, size, 0);  // Allocate pinned
 * cudaMemcpy(...);                // Fast transfer
 * cudaFreeHost(ptr);              // Free
 */

void comparePinnedVsPageable(int n) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Pinned vs Pageable Memory Transfer Speed\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    size_t bytes = n * sizeof(float);
    
    // Pageable memory
    float *h_pageable = (float*)malloc(bytes);
    
    // Pinned memory
    float *h_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes, 0));
    
    // Device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize both
    for (int i = 0; i < n; i++) {
        h_pageable[i] = (float)i;
        h_pinned[i] = (float)i;
    }
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test pageable memory
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 10; iter++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes,
                             cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pageable_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pageable_time, start, stop));
    
    // Test pinned memory
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 10; iter++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes,
                             cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pinned_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pinned_time, start, stop));
    
    float gb = (bytes * 10) / (1024.0f * 1024.0f * 1024.0f);
    
    printf("Results (10 transfers of %.2f MB):\n", 
           bytes / (1024.0f * 1024.0f));
    printf("┌──────────────┬──────────┬──────────────┐\n");
    printf("│ Memory Type  │ Time(ms) │ Bandwidth    │\n");
    printf("├──────────────┼──────────┼──────────────┤\n");
    printf("│ Pageable     │ %7.2f  │ %7.2f GB/s│\n", 
           pageable_time, gb / (pageable_time / 1000.0f));
    printf("│ Pinned       │ %7.2f  │ %7.2f GB/s│\n", 
           pinned_time, gb / (pinned_time / 1000.0f));
    printf("└──────────────┴──────────┴──────────────┘\n");
    printf("Speedup: %.2fx\n\n", pageable_time / pinned_time);
    
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   5. MEMORY ADVISE (Unified Memory Hints)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Provide hints to the CUDA runtime for better unified memory performance.
 *
 * Available Hints:
 * ───────────────
 * cudaMemAdviseSetReadMostly
 *   → Data will be read but rarely written
 *   → Runtime can create read-only copies on multiple devices
 *
 * cudaMemAdviseSetPreferredLocation
 *   → Data should preferably reside on specified device
 *   → Reduces migration overhead
 *
 * cudaMemAdviseSetAccessedBy
 *   → Data will be accessed by specified device
 *   → Can enable peer-to-peer access or direct access
 *
 * Example Use Case:
 * ────────────────
 * Large read-only lookup table used by all GPUs:
 *   cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, 0);
 *   → Runtime creates cached copies on each GPU
 *   → No migration needed
 */

void demonstrateMemoryAdvise(int n) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Memory Advise Example\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    size_t bytes = n * sizeof(float);
    
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    // Hint 1: This data will be read-mostly
    printf("Setting memory advice: READ_MOSTLY\n");
    CUDA_CHECK(cudaMemAdvise(data, bytes, 
                            cudaMemAdviseSetReadMostly, device));
    
    // Hint 2: Prefer this data on GPU
    printf("Setting memory advice: PREFERRED_LOCATION = GPU\n");
    CUDA_CHECK(cudaMemAdvise(data, bytes,
                            cudaMemAdviseSetPreferredLocation, device));
    
    // Hint 3: Will be accessed by GPU
    printf("Setting memory advice: ACCESSED_BY = GPU\n");
    CUDA_CHECK(cudaMemAdvise(data, bytes,
                            cudaMemAdviseSetAccessedBy, device));
    
    printf("\n✓ Memory hints set\n");
    printf("  Runtime can optimize based on these hints\n");
    printf("  Read-mostly data can be replicated\n");
    printf("  Preferred location reduces migrations\n\n");
    
    CUDA_CHECK(cudaFree(data));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   6. MEMORY POOLS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Memory pools reduce allocation overhead by reusing memory.
 *
 * Without Pool:              With Pool:
 * ─────────────             ──────────
 * cudaMalloc (slow)         cudaMallocAsync (fast)
 * kernel                    kernel
 * cudaFree (slow)           cudaFreeAsync (fast)
 * cudaMalloc (slow)         cudaMallocAsync (reuse!)
 * kernel                    kernel
 *
 * Benefits:
 * ────────
 * ✓ Faster allocation/deallocation
 * ✓ Reduced fragmentation
 * ✓ Stream-ordered operations
 */

void demonstrateMemoryPool(int n) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Memory Pool Example\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    size_t bytes = n * sizeof(float);
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int iterations = 100;
    
    // Test without pool (regular malloc/free)
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaFree(d_data));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float regular_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&regular_time, start, stop));
    
    // Test with pool (async malloc/free)
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        float *d_data;
        CUDA_CHECK(cudaMallocAsync(&d_data, bytes, stream));
        CUDA_CHECK(cudaFreeAsync(d_data, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float pool_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pool_time, start, stop));
    
    printf("Results (%d allocations + frees):\n", iterations);
    printf("┌──────────────────┬──────────┐\n");
    printf("│ Method           │ Time(ms) │\n");
    printf("├──────────────────┼──────────┤\n");
    printf("│ Regular malloc   │ %7.2f  │\n", regular_time);
    printf("│ Pool malloc      │ %7.2f  │\n", pool_time);
    printf("└──────────────────┴──────────┘\n");
    printf("Speedup: %.2fx\n\n", regular_time / pool_time);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║     CUDA Tutorial: Advanced Memory Techniques         ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    int n = 10000000;  // 10M elements = ~40 MB
    
    // Run demonstrations
    demonstrateZeroCopy(n);
    demonstrateUnifiedMemory(n);
    comparePinnedVsPageable(n);
    demonstrateMemoryAdvise(n);
    demonstrateMemoryPool(n);
    
    // Summary
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Texture memory: cached, filtered spatial access   ║\n");
    printf("║ 2. Zero-copy: direct host access, saves device mem   ║\n");
    printf("║ 3. Unified memory: automatic migration, easy to use  ║\n");
    printf("║ 4. Pinned memory: 2-4x faster transfers              ║\n");
    printf("║ 5. Memory pools: faster alloc/free in loops          ║\n");
    printf("║ 6. Choose technique based on access pattern          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    printf("Memory Technique Selection Guide:\n");
    printf("─────────────────────────────────\n");
    printf("• Spatial 2D/3D access    → Texture memory\n");
    printf("• One-time large data     → Zero-copy\n");
    printf("• Ease of programming     → Unified memory\n");
    printf("• Frequent host-device    → Pinned memory\n");
    printf("• Many alloc/free ops     → Memory pools\n");
    printf("• Read-only shared data   → Memory advise\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement 3D texture access for volume rendering
 * 2. Compare unified memory with/without prefetching
 * 3. Measure zero-copy performance vs. different problem sizes
 * 4. Implement double-buffering with pinned memory
 * 5. Create custom memory pool with different allocation strategies
 * 6. Benchmark memory advise impact on multi-GPU code
 * 7. Implement write-combined memory for faster GPU writes
 * 8. Compare texture vs. global memory for image convolution
 *
 * ═══════════════════════════════════════════════════════════════════
 */

