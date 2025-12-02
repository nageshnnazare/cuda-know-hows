/*
 * CUDA Tutorial - Part 19: Multi-GPU Programming
 * 
 * This file demonstrates multi-GPU programming techniques:
 * 1. Device Enumeration and Selection
 * 2. Peer-to-Peer (P2P) Memory Access
 * 3. Multi-GPU Data Parallelism
 * 4. Multi-GPU Matrix Multiplication
 * 5. GPU-Direct Communication
 * 6. Multi-Stream Multi-GPU
 * 7. Unified Memory with Multiple GPUs
 * 8. Load Balancing Strategies
 *
 * Each section includes:
 * - When to use multiple GPUs
 * - Communication patterns
 * - Performance considerations
 * - Best practices
 *
 * Compile: nvcc -o multi_gpu 19_multi_gpu.cu -O3
 * Run:     ./multi_gpu
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
 *              1. DEVICE ENUMERATION AND SELECTION
 * ═══════════════════════════════════════════════════════════════════
 *
 * First step: Discover available GPUs and their capabilities.
 *
 * Multi-GPU System Example:
 * ────────────────────────
 *   CPU
 *    │
 *    ├─── PCIe ─── GPU 0 (RTX 3090, 24 GB)
 *    ├─── PCIe ─── GPU 1 (RTX 3090, 24 GB)
 *    ├─── PCIe ─── GPU 2 (RTX 3080, 10 GB)
 *    └─── PCIe ─── GPU 3 (RTX 3080, 10 GB)
 *
 * Device Selection:
 * ────────────────
 * cudaSetDevice(device_id) - Set current device for this host thread
 * cudaGetDevice(&device_id) - Get current device
 *
 * Important: Each host thread has its own current device!
 *
 * Thread Management:
 * ─────────────────
 * Host Thread 0 → cudaSetDevice(0) → Works on GPU 0
 * Host Thread 1 → cudaSetDevice(1) → Works on GPU 1
 * Host Thread 2 → cudaSetDevice(2) → Works on GPU 2
 * Host Thread 3 → cudaSetDevice(3) → Works on GPU 3
 */

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU %d: %s\n", device, prop.name);
    printf("  Compute Capability:      %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory:     %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors:         %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block:   %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per SM:      %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Memory Clock Rate:       %.2f GHz\n", 
           prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width:        %d-bit\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth:   %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Peer-to-Peer Support:    %s\n",
           prop.unifiedAddressing ? "Yes" : "No");
    printf("\n");
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              2. PEER-TO-PEER (P2P) MEMORY ACCESS
 * ═══════════════════════════════════════════════════════════════════
 *
 * P2P allows GPUs to directly access each other's memory without
 * going through host memory.
 *
 * Without P2P:                    With P2P:
 * ────────────                    ─────────
 * GPU 0 → Host Memory → GPU 1     GPU 0 ←→ GPU 1
 *   ↓         ↓                     (Direct!)
 * Slow      Slow                    Fast
 *
 * Visual Data Transfer:
 * ────────────────────
 * Traditional (3 copies):
 * ┌──────┐    ┌──────┐    ┌──────┐
 * │ GPU 0│ →  │ Host │ →  │ GPU 1│
 * └──────┘    └──────┘    └──────┘
 *    2 GB       2 GB        2 GB
 *
 * P2P (1 copy):
 * ┌──────┐ ─────────────→ ┌──────┐
 * │ GPU 0│                 │ GPU 1│
 * └──────┘                 └──────┘
 *    2 GB                     2 GB
 *
 * Topology Considerations:
 * ───────────────────────
 * Same PCIe Switch: Fast (~10 GB/s)
 * Different Switch:  Slower (~5 GB/s)
 * NVLink:           Very Fast (~50 GB/s per link)
 *
 * Requirements:
 * ────────────
 * • Compute capability ≥ 2.0
 * • Both GPUs on same PCIe root complex (usually)
 * • cudaDeviceCanAccessPeer() returns true
 */

bool canAccessPeer(int srcDevice, int dstDevice) {
    int canAccess = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, srcDevice, dstDevice));
    return canAccess != 0;
}

void enableP2P(int device1, int device2) {
    if (canAccessPeer(device1, device2)) {
        CUDA_CHECK(cudaSetDevice(device1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(device2, 0));
        printf("✓ Enabled P2P: GPU %d → GPU %d\n", device1, device2);
    } else {
        printf("✗ P2P not available: GPU %d → GPU %d\n", device1, device2);
    }
}

// Kernel that runs on GPU srcDevice but accesses memory from GPU dstDevice
__global__ void p2pKernel(float *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct access to peer GPU's memory!
        dst[idx] = src[idx] * 2.0f;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              3. MULTI-GPU DATA PARALLELISM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Most common pattern: Split data across GPUs, process in parallel.
 *
 * Single GPU:                     Multi-GPU (4 GPUs):
 * ──────────                      ───────────────────
 * ┌────────────┐                 ┌───┐ ┌───┐ ┌───┐ ┌───┐
 * │            │                 │ ¼ │ │ ¼ │ │ ¼ │ │ ¼ │
 * │  All Data  │    →            │   │ │   │ │   │ │   │
 * │            │                 │   │ │   │ │   │ │   │
 * └────────────┘                 └───┘ └───┘ └───┘ └───┘
 *   Process all                  GPU0  GPU1  GPU2  GPU3
 *   sequentially                 Process in parallel!
 *
 * Steps:
 * ─────
 * 1. Divide data into N chunks (N = number of GPUs)
 * 2. Allocate memory on each GPU
 * 3. Copy chunk to each GPU
 * 4. Launch kernel on each GPU
 * 5. Copy results back
 * 6. Combine results
 *
 * Example: Vector Addition
 * ───────────────────────
 * Total: 8M elements
 * GPU 0: elements 0-2M      ┐
 * GPU 1: elements 2M-4M     │ Process
 * GPU 2: elements 4M-6M     │ in
 * GPU 3: elements 6M-8M     ┘ parallel
 *
 * Ideal Speedup: 4x (with 4 GPUs)
 * Real Speedup:  3.5x (communication overhead)
 */

void vectorAddMultiGPU(float *h_a, float *h_b, float *h_c, 
                       int total_size, int num_gpus) {
    int chunk_size = (total_size + num_gpus - 1) / num_gpus;
    
    // Arrays to store per-GPU pointers
    float **d_a = (float**)malloc(num_gpus * sizeof(float*));
    float **d_b = (float**)malloc(num_gpus * sizeof(float*));
    float **d_c = (float**)malloc(num_gpus * sizeof(float*));
    
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    
    // Allocate and copy data to each GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        
        int offset = gpu * chunk_size;
        int current_chunk = min(chunk_size, total_size - offset);
        size_t bytes = current_chunk * sizeof(float);
        
        // Allocate on this GPU
        CUDA_CHECK(cudaMalloc(&d_a[gpu], bytes));
        CUDA_CHECK(cudaMalloc(&d_b[gpu], bytes));
        CUDA_CHECK(cudaMalloc(&d_c[gpu], bytes));
        
        // Copy data to this GPU (asynchronously)
        CUDA_CHECK(cudaMemcpyAsync(d_a[gpu], h_a + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[gpu]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[gpu], h_b + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[gpu]));
    }
    
    // Launch kernels on all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        int offset = gpu * chunk_size;
        int current_chunk = min(chunk_size, total_size - offset);
        
        int blockSize = 256;
        int gridSize = (current_chunk + blockSize - 1) / blockSize;
        
        // Launch on this GPU's stream
        // (Assuming vectorAddKernel is defined)
        // vectorAddKernel<<<gridSize, blockSize, 0, streams[gpu]>>>
        //     (d_a[gpu], d_b[gpu], d_c[gpu], current_chunk);
    }
    
    // Copy results back from all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        int offset = gpu * chunk_size;
        int current_chunk = min(chunk_size, total_size - offset);
        size_t bytes = current_chunk * sizeof(float);
        
        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c[gpu], bytes,
                                   cudaMemcpyDeviceToHost, streams[gpu]));
    }
    
    // Synchronize all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    // Cleanup
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaFree(d_a[gpu]));
        CUDA_CHECK(cudaFree(d_b[gpu]));
        CUDA_CHECK(cudaFree(d_c[gpu]));
        CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
    }
    
    free(d_a);
    free(d_b);
    free(d_c);
    free(streams);
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              4. MULTI-GPU MATRIX MULTIPLICATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Matrix multiplication with row-wise distribution.
 *
 * Matrix A (M×K):                Split across GPUs:
 * ┌──────────────┐              ┌──────────────┐ GPU 0
 * │              │              ├──────────────┤
 * │              │              ├──────────────┤ GPU 1
 * │              │      →       ├──────────────┤
 * │              │              ├──────────────┤ GPU 2
 * │              │              └──────────────┘
 * └──────────────┘              Each GPU gets M/N rows
 *
 * Matrix B (K×N): Replicated on all GPUs (needed by all)
 *
 * Algorithm:
 * ─────────
 * 1. Distribute rows of A across GPUs
 * 2. Replicate B on all GPUs
 * 3. Each GPU computes its portion of C
 * 4. Gather results
 *
 * Communication:
 * ─────────────
 * Broadcast B:     1× per GPU
 * Scatter A:       1× per GPU
 * Gather C:        1× per GPU
 *
 * Computation/Communication Ratio:
 * ────────────────────────────────
 * Computation: O(M×K×N / num_gpus)
 * Communication: O((M+K×N) / num_gpus + K×N)
 *
 * Good for large matrices where computation >> communication
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *              5. UNIFIED MEMORY WITH MULTIPLE GPUS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Unified Memory automatically migrates between GPUs.
 *
 * Without Unified Memory:
 * ──────────────────────
 * • Manual memory management
 * • Explicit copies between GPUs
 * • Complex bookkeeping
 *
 * With Unified Memory:
 * ───────────────────
 * • Single pointer works on all GPUs
 * • Automatic migration
 * • Page faults trigger transfers
 *
 * Visual:
 * ──────
 * cudaMallocManaged(&ptr, size);
 *
 * GPU 0 access → Data migrates to GPU 0
 * GPU 1 access → Data migrates to GPU 1
 * CPU access   → Data migrates to CPU
 *
 * Optimization:
 * ────────────
 * cudaMemPrefetchAsync(ptr, size, device);  // Hint to runtime
 * cudaMemAdviseSetPreferredLocation(ptr, size, device);
 * cudaMemAdviseSetAccessedBy(ptr, size, device);
 */

void unifiedMemoryMultiGPU(int num_gpus) {
    size_t size = 1024 * 1024 * 100;  // 100M elements
    float *unified_ptr;
    
    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&unified_ptr, size * sizeof(float)));
    
    // Initialize on CPU
    for (size_t i = 0; i < size; i++) {
        unified_ptr[i] = (float)i;
    }
    
    // Process on multiple GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        // Optional: Prefetch to this GPU
        size_t chunk_start = (size / num_gpus) * gpu;
        size_t chunk_size = size / num_gpus;
        
        CUDA_CHECK(cudaMemPrefetchAsync(unified_ptr + chunk_start,
                                        chunk_size * sizeof(float),
                                        gpu));
        
        // Launch kernel that accesses unified_ptr
        // Data automatically migrates as needed
    }
    
    // Synchronize all
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    CUDA_CHECK(cudaFree(unified_ptr));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              6. LOAD BALANCING STRATEGIES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Problem: Not all GPUs are equally fast!
 *
 * Static Division (Bad):
 * ─────────────────────
 * GPU 0 (Fast):   ████████░░  80% time idle
 * GPU 1 (Slow):   ██████████  100% utilized
 *
 * Dynamic Division (Good):
 * ───────────────────────
 * GPU 0 (Fast):   ████████████  60% of work
 * GPU 1 (Slow):   ████████      40% of work
 *
 * Strategies:
 * ──────────
 *
 * 1. Proportional to Performance:
 *    work[i] = total_work × (perf[i] / Σperf[j])
 *
 * 2. Work Stealing:
 *    • Each GPU has a work queue
 *    • Idle GPU steals from others
 *
 * 3. Dynamic Scheduling:
 *    • Start with small chunks
 *    • GPU requests more when done
 *
 * Example: 1000 tasks, 2 GPUs
 * ──────────────────────────
 * GPU 0 (2x faster): gets 667 tasks
 * GPU 1 (1x faster): gets 333 tasks
 * Finish at same time!
 */

void balancedMultiGPU(int num_gpus) {
    // Get performance of each GPU
    float *gpu_performance = (float*)malloc(num_gpus * sizeof(float));
    float total_performance = 0.0f;
    
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
        
        // Estimate relative performance
        // (simplified: use SM count × clock rate)
        gpu_performance[gpu] = prop.multiProcessorCount * prop.clockRate;
        total_performance += gpu_performance[gpu];
    }
    
    // Distribute work proportionally
    int total_work = 10000000;  // 10M units of work
    
    printf("Load Balanced Work Distribution:\n");
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        float fraction = gpu_performance[gpu] / total_performance;
        int assigned_work = (int)(total_work * fraction);
        
        printf("  GPU %d: %d units (%.1f%%)\n", 
               gpu, assigned_work, fraction * 100.0f);
    }
    
    free(gpu_performance);
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              7. MULTI-GPU BEST PRACTICES
 * ═══════════════════════════════════════════════════════════════════
 *
 * DO:
 * ──
 * ✓ Use streams for overlapping communication and computation
 * ✓ Enable P2P when available
 * ✓ Balance load based on GPU performance
 * ✓ Minimize data transfers between GPUs
 * ✓ Use pinned memory for fast host-device transfers
 * ✓ Prefetch unified memory to target device
 * ✓ Check for errors on all devices
 *
 * DON'T:
 * ─────
 * ✗ Assume all GPUs are identical
 * ✗ Use cudaDeviceSynchronize() unnecessarily (blocks all streams)
 * ✗ Forget to set device before each CUDA call
 * ✗ Ignore topology (PCIe vs NVLink)
 * ✗ Copy large data between GPUs frequently
 * ✗ Use blocking copies (use async)
 *
 * Communication Patterns:
 * ──────────────────────
 *
 * 1. All-Reduce (Sum across GPUs):
 *    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
 *    │  5   │   │  3   │   │  7   │   │  2   │
 *    └───┬──┘   └───┬──┘   └───┬──┘   └───┬──┘
 *        │          │          │          │
 *        └──────────┴──────────┴──────────┘
 *                     ↓
 *                Sum = 17 on all GPUs
 *
 * 2. Broadcast:
 *    ┌──────┐
 *    │ GPU 0│ → Data
 *    └──┬───┘
 *       ├────→ GPU 1
 *       ├────→ GPU 2
 *       └────→ GPU 3
 *
 * 3. Scatter:
 *    ┌──────────────┐
 *    │ All Data     │
 *    └──┬───────────┘
 *       ├─ Chunk 0 → GPU 0
 *       ├─ Chunk 1 → GPU 1
 *       ├─ Chunk 2 → GPU 2
 *       └─ Chunk 3 → GPU 3
 *
 * 4. Gather:
 *    GPU 0 → Chunk 0 ─┐
 *    GPU 1 → Chunk 1 ─┤
 *    GPU 2 → Chunk 2 ─┼→ ┌──────────────┐
 *    GPU 3 → Chunk 3 ─┘  │ Combined Data│
 *                        └──────────────┘
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Tutorial: Multi-GPU Programming          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Get number of GPUs
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    
    if (num_gpus < 1) {
        printf("No CUDA-capable devices found!\n");
        return EXIT_FAILURE;
    }
    
    printf("Found %d CUDA device(s)\n\n", num_gpus);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Device Properties
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Device Properties\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    for (int i = 0; i < num_gpus; i++) {
        printDeviceProperties(i);
    }
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Peer-to-Peer Access
     * ───────────────────────────────────────────────────────────────
     */
    
    if (num_gpus >= 2) {
        printf("═══════════════════════════════════════════════════════\n");
        printf("Test 2: Peer-to-Peer Access\n");
        printf("═══════════════════════════════════════════════════════\n\n");
        
        printf("P2P Access Matrix:\n");
        printf("     ");
        for (int j = 0; j < num_gpus; j++) {
            printf("GPU%d ", j);
        }
        printf("\n");
        
        for (int i = 0; i < num_gpus; i++) {
            printf("GPU%d ", i);
            for (int j = 0; j < num_gpus; j++) {
                if (i == j) {
                    printf("  -  ");
                } else {
                    bool can_access = canAccessPeer(i, j);
                    printf("  %s  ", can_access ? "✓" : "✗");
                }
            }
            printf("\n");
        }
        printf("\n");
        
        // Try to enable P2P between first two GPUs
        if (num_gpus >= 2) {
            printf("Enabling P2P between GPU 0 and GPU 1:\n");
            enableP2P(0, 1);
            enableP2P(1, 0);
            printf("\n");
        }
    }
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Load Balancing
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Load Balancing\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    balancedMultiGPU(num_gpus);
    printf("\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Use cudaSetDevice() to select active GPU          ║\n");
    printf("║ 2. Enable P2P for direct GPU-to-GPU transfers        ║\n");
    printf("║ 3. Balance load based on GPU performance             ║\n");
    printf("║ 4. Use streams for overlapping operations            ║\n");
    printf("║ 5. Consider topology (PCIe, NVLink)                  ║\n");
    printf("║ 6. Minimize inter-GPU communication                  ║\n");
    printf("║ 7. Unified Memory simplifies multi-GPU code          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    printf("Multi-GPU Scalability:\n");
    printf("──────────────────────\n");
    printf("Ideal Speedup:      N× (N = number of GPUs)\n");
    printf("Typical Speedup:    0.8N× to 0.95N× (communication overhead)\n");
    printf("Best Use Cases:     Data parallel workloads\n");
    printf("                    Large matrix operations\n");
    printf("                    Deep learning training\n");
    printf("                    Scientific simulations\n\n");
    
    printf("When to Use Multi-GPU:\n");
    printf("─────────────────────\n");
    printf("✓ Problem too large for single GPU memory\n");
    printf("✓ Computation >> Communication\n");
    printf("✓ Embarrassingly parallel workload\n");
    printf("✓ Need higher throughput\n\n");
    
    printf("When NOT to Use Multi-GPU:\n");
    printf("─────────────────────────\n");
    printf("✗ High inter-GPU communication needed\n");
    printf("✗ Sequential dependencies\n");
    printf("✗ Small problem size\n");
    printf("✗ Single GPU sufficient\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement multi-GPU matrix multiplication
 * 2. Add work-stealing load balancer
 * 3. Implement all-reduce operation
 * 4. Benchmark P2P vs through-host transfers
 * 5. Add multi-GPU deep learning training
 * 6. Implement multi-GPU sorting
 * 7. Add topology-aware communication
 * 8. Create multi-GPU stream pool
 *
 * Advanced:
 * ────────
 * 9. Implement NCCL-like collectives
 * 10. Add GPU-Direct RDMA support
 * 11. Implement pipeline parallelism
 * 12. Add dynamic resource allocation
 *
 * ═══════════════════════════════════════════════════════════════════
 */

