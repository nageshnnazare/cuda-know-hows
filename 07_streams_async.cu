/*
 * CUDA Tutorial - Part 7: Streams and Asynchronous Operations
 * 
 * This file demonstrates:
 * 1. CUDA streams for concurrent execution
 * 2. Asynchronous memory transfers
 * 3. Overlapping computation and communication
 * 4. Multi-GPU programming basics
 * 5. Performance optimization with streams
 *
 * Compile: nvcc -o streams 07_streams_async.cu
 * Run:     ./streams
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
 *                        CUDA STREAMS OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * What is a CUDA Stream?
 * ─────────────────────
 * A sequence of operations that execute in order on the GPU.
 * Operations in different streams can execute concurrently.
 *
 * Default Stream (NULL stream):
 * ────────────────────────────
 * - Synchronizes with all other streams
 * - Used when no stream is specified
 * - Can block other operations
 *
 * Non-default Streams:
 * ───────────────────
 * - Can execute concurrently
 * - Enable overlap of operations
 * - Require explicit creation
 *
 * Typical Execution Timeline:
 * ──────────────────────────
 *
 * WITHOUT STREAMS (Sequential):
 * ─────────────────────────────────────────────────────────
 * CPU: |H2D|Kernel|D2H|H2D|Kernel|D2H|H2D|Kernel|D2H|
 * GPU:     |Kernel|   |Kernel|   |Kernel|
 *
 * WITH STREAMS (Concurrent):
 * ─────────────────────────────────────────────────────────
 * Stream 0: |H2D|Kernel|D2H|
 * Stream 1:     |H2D|Kernel|D2H|
 * Stream 2:         |H2D|Kernel|D2H|
 * 
 * Timeline:
 * ═════════╪═══════╪═══════╪═══════╪═══════╪═══════╪═════
 * Copy     │ S0→   │ S1→   │ S2→   │       │       │
 * Engine   │       │       │       │       │       │
 * ─────────┼───────┼───────┼───────┼───────┼───────┼─────
 * Compute  │       │  S0   │  S1   │  S2   │       │
 * Engine   │       │       │       │       │       │
 * ─────────┼───────┼───────┼───────┼───────┼───────┼─────
 * Copy     │       │       │  ←S0  │  ←S1  │  ←S2  │
 * Engine   │       │       │       │       │       │
 * ═════════╧═══════╧═══════╧═══════╧═══════╧═══════╧═════
 *
 * Benefits:
 * ────────
 * • Hide memory transfer latency
 * • Overlap computation and communication
 * • Better GPU utilization
 * • Enable concurrent kernel execution
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    SIMPLE KERNEL FOR TESTING
 * ═══════════════════════════════════════════════════════════════════
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Add some computation to make it more realistic
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            sum += sinf(a[idx]) * cosf(b[idx]);
        }
        c[idx] = a[idx] + b[idx] + sum * 0.0001f;
    }
}

__global__ void vectorScale(float *a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 1: BASIC STREAMS
 * ═══════════════════════════════════════════════════════════════════
 */

void demonstrateBasicStreams() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 1: Basic Stream Operations\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    // Allocate pinned host memory (required for async transfers)
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Create streams
    const int nStreams = 4;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test 1: Without streams (synchronous)
    printf("Test 1: Synchronous execution (no streams)\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float syncTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&syncTime, start, stop));
    printf("  Time: %.3f ms\n\n", syncTime);
    
    // Test 2: With streams (asynchronous)
    printf("Test 2: Asynchronous execution with %d streams\n", nStreams);
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch operations in multiple streams
    for (int i = 0; i < nStreams; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, streams[i]));
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float asyncTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&asyncTime, start, stop));
    printf("  Time: %.3f ms\n", asyncTime);
    printf("  Speedup: %.2fx\n\n", syncTime / asyncTime);
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              EXAMPLE 2: OVERLAPPING COMPUTE AND TRANSFER
 * ═══════════════════════════════════════════════════════════════════
 *
 * Strategy: Divide data into chunks and pipeline:
 * 
 * Chunk 0: |H2D|Compute|D2H|
 * Chunk 1:     |H2D|Compute|D2H|
 * Chunk 2:         |H2D|Compute|D2H|
 * Chunk 3:             |H2D|Compute|D2H|
 *
 * This hides transfer latency behind computation!
 */

void demonstratePipelining() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 2: Pipelining with Streams\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 22;  // 4M elements
    const int nStreams = 4;
    const int chunkSize = N / nStreams;
    size_t chunkBytes = chunkSize * sizeof(float);
    
    printf("Total elements: %d\n", N);
    printf("Number of streams: %d\n", nStreams);
    printf("Elements per stream: %d\n\n", chunkSize);
    
    // Allocate pinned host memory
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i * 0.01f);
        h_b[i] = cosf(i * 0.01f);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    // Create streams
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int blockSize = 256;
    int gridSize = (chunkSize + blockSize - 1) / blockSize;
    
    // Pipelined execution
    printf("Executing pipelined operations...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunkSize;
        
        // Copy chunk to device
        CUDA_CHECK(cudaMemcpyAsync(&d_a[offset], &h_a[offset], chunkBytes,
                                   cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(&d_b[offset], &h_b[offset], chunkBytes,
                                   cudaMemcpyHostToDevice, streams[i]));
        
        // Compute on chunk
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>
            (&d_a[offset], &d_b[offset], &d_c[offset], chunkSize);
        
        // Copy result back
        CUDA_CHECK(cudaMemcpyAsync(&h_c[offset], &d_c[offset], chunkBytes,
                                   cudaMemcpyDeviceToHost, streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pipelineTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pipelineTime, start, stop));
    
    printf("Pipeline execution time: %.3f ms\n", pipelineTime);
    printf("Throughput: %.2f GB/s\n\n",
           (3.0f * N * sizeof(float)) / (pipelineTime / 1000.0f) / 1e9f);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N && i < 100; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n\n", correct ? "✓ PASSED" : "✗ FAILED");
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              EXAMPLE 3: STREAM SYNCHRONIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Different ways to synchronize streams:
 * 1. cudaStreamSynchronize(stream) - wait for specific stream
 * 2. cudaDeviceSynchronize() - wait for all streams
 * 3. cudaStreamWaitEvent() - stream waits for event
 * 4. cudaEventSynchronize() - host waits for event
 */

void demonstrateSynchronization() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 3: Stream Synchronization\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    cudaEvent_t event1, event2;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Demonstrating stream dependencies:\n\n");
    
    // Stream 1: Scale by 2
    vectorScale<<<gridSize, blockSize, 0, stream1>>>(d_a, 2.0f, N);
    CUDA_CHECK(cudaEventRecord(event1, stream1));
    printf("  Stream 1: Scaling array by 2.0\n");
    
    // Stream 2 waits for Stream 1
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event1, 0));
    vectorScale<<<gridSize, blockSize, 0, stream2>>>(d_a, 3.0f, N);
    CUDA_CHECK(cudaEventRecord(event2, stream2));
    printf("  Stream 2: Waiting for Stream 1, then scaling by 3.0\n");
    
    // Synchronize
    CUDA_CHECK(cudaEventSynchronize(event2));
    printf("  Result: Array scaled by 2.0 × 3.0 = 6.0\n\n");
    
    printf("Synchronization methods:\n");
    printf("  1. cudaStreamSynchronize(stream) - Wait for one stream\n");
    printf("  2. cudaDeviceSynchronize()       - Wait for all operations\n");
    printf("  3. cudaStreamWaitEvent()         - Inter-stream dependency\n");
    printf("  4. cudaEventSynchronize()        - Wait for specific event\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    EXAMPLE 4: MULTI-GPU BASICS
 * ═══════════════════════════════════════════════════════════════════
 */

void demonstrateMultiGPU() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 4: Multi-GPU Programming Basics\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Multi-GPU example requires at least 2 GPUs.\n");
        printf("Showing single GPU information only.\n\n");
    }
    
    for (int i = 0; i < deviceCount && i < 4; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Concurrent Kernels: %s\n", 
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Async Engine Count: %d\n\n", prop.asyncEngineCount);
    }
    
    if (deviceCount >= 2) {
        printf("Multi-GPU execution pattern:\n");
        printf("────────────────────────────\n");
        printf("  1. cudaSetDevice(0) - Select GPU 0\n");
        printf("  2. Allocate memory on GPU 0\n");
        printf("  3. Launch kernel on GPU 0\n");
        printf("  4. cudaSetDevice(1) - Select GPU 1\n");
        printf("  5. Allocate memory on GPU 1\n");
        printf("  6. Launch kernel on GPU 1\n");
        printf("  7. Synchronize both GPUs\n\n");
        
        printf("Advanced: Peer-to-peer transfers\n");
        printf("─────────────────────────────────\n");
        
        int canAccessPeer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
        printf("  GPU 0 can access GPU 1: %s\n", 
               canAccessPeer ? "Yes (P2P enabled)" : "No");
        
        if (canAccessPeer) {
            printf("  Benefits: Direct GPU-to-GPU transfers\n");
            printf("  Usage: cudaMemcpyPeer()\n");
        }
    }
    printf("\n");
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║    CUDA Tutorial: Streams & Async Operations         ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("📊 Device: %s\n", prop.name);
    printf("   Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "✓ Supported" : "✗ Not supported");
    printf("   Async engine count: %d\n", prop.asyncEngineCount);
    printf("   Can map host memory: %s\n\n",
           prop.canMapHostMemory ? "✓ Yes" : "✗ No");
    
    if (!prop.concurrentKernels) {
        printf("⚠️  Warning: This device does not support concurrent kernels.\n");
        printf("   Stream benefits will be limited.\n\n");
    }
    
    // Run examples
    demonstrateBasicStreams();
    demonstratePipelining();
    demonstrateSynchronization();
    demonstrateMultiGPU();
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Best Practices Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Streams Best Practices\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("✓ DO:\n");
    printf("  • Use pinned memory (cudaMallocHost) for async transfers\n");
    printf("  • Overlap data transfers with kernel execution\n");
    printf("  • Use multiple streams for independent operations\n");
    printf("  • Consider work granularity vs overhead\n");
    printf("  • Profile to verify actual concurrency\n\n");
    
    printf("✗ DON'T:\n");
    printf("  • Use streams if operations have dependencies\n");
    printf("  • Create too many streams (overhead increases)\n");
    printf("  • Forget to destroy streams (memory leak)\n");
    printf("  • Use default stream if you need concurrency\n");
    printf("  • Assume automatic optimization (measure!)\n\n");
    
    printf("MEMORY TYPES FOR ASYNC:\n");
    printf("───────────────────────────────────────────────────────\n");
    printf("  Type              Async Transfer    Performance\n");
    printf("  ────────────────────────────────────────────────────\n");
    printf("  Pageable (malloc) ✗ No             Standard\n");
    printf("  Pinned (cudaMallocHost) ✓ Yes      ~2x faster\n");
    printf("  Mapped            ✓ Yes            Special use\n");
    printf("  Managed (Unified) ✓ Yes            Convenient\n\n");
    
    printf("STREAM PRIORITY:\n");
    printf("───────────────────────────────────────────────────────\n");
    printf("  cudaStreamCreateWithPriority() allows setting priority\n");
    printf("  Higher priority streams get more resources\n");
    printf("  Useful for latency-critical operations\n\n");
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Streams enable concurrent GPU operations          ║\n");
    printf("║ 2. Pinned memory is required for async transfers     ║\n");
    printf("║ 3. Pipeline data transfers with computation          ║\n");
    printf("║ 4. Use events for fine-grained synchronization       ║\n");
    printf("║ 5. Profile to verify actual performance gains        ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement a producer-consumer pattern with streams
 * 2. Create a multi-GPU matrix multiplication
 * 3. Optimize image processing pipeline with streams
 * 4. Implement double buffering for continuous processing
 * 5. Measure actual overlap using NVIDIA Visual Profiler
 * 6. Compare pinned vs pageable memory performance
 *
 * ═══════════════════════════════════════════════════════════════════
 */

