/*
 * CUDA Tutorial - Part 2: Your First CUDA Kernel
 * 
 * This file demonstrates:
 * 1. Writing a simple CUDA kernel
 * 2. Memory allocation and transfer
 * 3. Kernel launch syntax
 * 4. Error handling
 * 5. Vector addition example
 *
 * Compile: nvcc -o first_kernel 02_first_kernel.cu
 * Run:     ./first_kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
 * ═══════════════════════════════════════════════════════════════════
 *                        ERROR HANDLING MACRO
 * ═══════════════════════════════════════════════════════════════════
 * 
 * Always check CUDA API calls for errors! This macro helps catch
 * and report errors with file and line information.
 */

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
 *                      EXAMPLE 1: HELLO FROM GPU
 * ═══════════════════════════════════════════════════════════════════
 *
 * The simplest possible kernel - just prints from each thread.
 * 
 * Execution Flow:
 * ┌─────────┐
 * │  HOST   │ launches kernel
 * │  (CPU)  │────────────────────────────┐
 * └─────────┘                            │
 *                                        ↓
 *                              ┌──────────────────┐
 *                              │   DEVICE (GPU)   │
 *                              │                  │
 *                              │  Thread 0 prints │
 *                              │  Thread 1 prints │
 *                              │  Thread 2 prints │
 *                              │      ...         │
 *                              └──────────────────┘
 *
 * __global__ = This function runs on GPU, called from CPU
 */

__global__ void helloFromGPU() {
    // Each thread calculates its unique global ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Hello from thread %d in block %d (global thread %d)\n", 
           threadIdx.x, blockIdx.x, tid);
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   EXAMPLE 2: VECTOR ADDITION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Vector addition: C[i] = A[i] + B[i]
 *
 * Sequential CPU version would look like:
 *   for (i = 0; i < N; i++)
 *       C[i] = A[i] + B[i];
 *
 * CUDA parallelizes this - each thread handles ONE element!
 *
 * Data Distribution Example (N=8, 2 blocks, 4 threads each):
 * 
 *   Array A: [a0][a1][a2][a3][a4][a5][a6][a7]
 *   Array B: [b0][b1][b2][b3][b4][b5][b6][b7]
 *            ─────────────────────────────────
 *   Block 0:  t0  t1  t2  t3
 *   Block 1:                  t0  t1  t2  t3
 *            ─────────────────────────────────
 *   Result:  [c0][c1][c2][c3][c4][c5][c6][c7]
 *
 * Each thread computes: result[tid] = a[tid] + b[tid]
 */

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    // Formula: global_id = block_id * threads_per_block + thread_id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: make sure we don't access beyond array bounds
    // This is necessary when N is not a perfect multiple of block size
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
        
        // Optional: print first few results for verification
        if (tid < 10) {
            printf("GPU: C[%d] = %.2f + %.2f = %.2f\n", 
                   tid, a[tid], b[tid], c[tid]);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   CPU VERSION (For Comparison)
 * ═══════════════════════════════════════════════════════════════════
 */

void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                        VERIFICATION FUNCTION
 * ═══════════════════════════════════════════════════════════════════
 */

bool verifyResults(const float *a, const float *b, const float *c, int n) {
    const float epsilon = 1e-5;
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > epsilon) {
            printf("Mismatch at index %d: expected %.5f, got %.5f\n", 
                   i, expected, c[i]);
            return false;
        }
    }
    return true;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║       CUDA Tutorial: First Kernel Examples            ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * STEP 1: Query GPU Information
     * ───────────────────────────────────────────────────────────────
     */
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable GPU found!\n");
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("📊 Device Information:\n");
    printf("   Name: %s\n", prop.name);
    printf("   Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("   Total Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("   Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("   Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("   Max Threads Dimensions: [%d, %d, %d]\n\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 1: Hello from GPU
     * ───────────────────────────────────────────────────────────────
     *
     * Launch configuration:
     *   <<<number_of_blocks, threads_per_block>>>
     *
     * Here we launch 2 blocks with 4 threads each = 8 total threads
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 1: Hello from GPU\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int numBlocks = 2;
    int threadsPerBlock = 4;
    
    printf("Launching %d blocks with %d threads each (%d total threads)\n\n", 
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);
    
    // Kernel launch syntax: kernel<<<blocks, threads>>>(args)
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();
    
    // Wait for GPU to finish (important for printf output)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for any errors during kernel execution
    CUDA_CHECK(cudaGetLastError());
    
    printf("\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 2: Vector Addition
     * ───────────────────────────────────────────────────────────────
     *
     * Memory Flow Diagram:
     * 
     *   HOST                              DEVICE
     *   ════                              ══════
     *   
     *   h_a ──────cudaMemcpy────────────→ d_a
     *   h_b ──────cudaMemcpy────────────→ d_b
     *   h_c                                d_c
     *                                       ↓
     *                              [GPU Computation]
     *                                       ↓
     *   h_c ←─────cudaMemcpy──────────── d_c
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 2: Vector Addition\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Vector size
    const int N = 1 << 20;  // 1M elements (2^20)
    size_t bytes = N * sizeof(float);
    
    printf("Vector size: %d elements (%.2f MB)\n\n", N, bytes / (1024.0 * 1024.0));
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 2: Allocate Host Memory
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("Step 1: Allocating host memory...\n");
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return EXIT_FAILURE;
    }
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 3: Initialize Host Data
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("Step 2: Initializing input data...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 4: Allocate Device Memory
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("Step 3: Allocating device memory...\n");
    float *d_a, *d_b, *d_c;
    
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 5: Copy Data from Host to Device
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("Step 4: Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 6: Launch Kernel
     * ─────────────────────────────────────────────────────────────
     *
     * Choosing block size and grid size:
     * - Block size: typically 128, 256, or 512 threads
     * - Grid size: (N + blockSize - 1) / blockSize
     *   This ensures we have enough threads to cover all elements
     */
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Step 5: Launching kernel with %d blocks of %d threads...\n", 
           gridSize, blockSize);
    printf("        Total threads: %d\n\n", gridSize * blockSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch the kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("\n⏱️  Kernel execution time: %.3f ms\n", milliseconds);
    printf("   Throughput: %.2f GB/s\n", 
           (3 * bytes) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0));
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 7: Copy Results from Device to Host
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("\nStep 6: Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 8: Verify Results
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("Step 7: Verifying results...\n");
    bool success = verifyResults(h_a, h_b, h_c, N);
    
    if (success) {
        printf("✓ Results verified successfully!\n");
        printf("  Example: %.5f + %.5f = %.5f\n", h_a[0], h_b[0], h_c[0]);
        printf("  Example: %.5f + %.5f = %.5f\n", h_a[100], h_b[100], h_c[100]);
    } else {
        printf("✗ Verification failed!\n");
    }
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 9: Compare with CPU Version
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("\nStep 8: Comparing with CPU version...\n");
    
    // Time CPU version
    cudaEvent_t cpuStart, cpuStop;
    CUDA_CHECK(cudaEventCreate(&cpuStart));
    CUDA_CHECK(cudaEventCreate(&cpuStop));
    
    CUDA_CHECK(cudaEventRecord(cpuStart));
    vectorAddCPU(h_a, h_b, h_c, N);
    CUDA_CHECK(cudaEventRecord(cpuStop));
    CUDA_CHECK(cudaEventSynchronize(cpuStop));
    
    float cpuTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cpuTime, cpuStart, cpuStop));
    
    printf("   CPU time: %.3f ms\n", cpuTime);
    printf("   GPU time: %.3f ms\n", milliseconds);
    printf("   Speedup: %.2fx\n", cpuTime / milliseconds);
    
    /*
     * ─────────────────────────────────────────────────────────────
     * STEP 10: Cleanup
     * ─────────────────────────────────────────────────────────────
     */
    
    printf("\nStep 9: Cleaning up...\n");
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(cpuStart));
    CUDA_CHECK(cudaEventDestroy(cpuStop));
    
    // Reset device
    CUDA_CHECK(cudaDeviceReset());
    
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf(  "║                Program completed successfully!        ║\n");
    printf(  "╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         KEY TAKEAWAYS
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. KERNEL SYNTAX:
 *    __global__ void kernelName(...) { }
 *    Launch: kernelName<<<blocks, threads>>>(args)
 *
 * 2. THREAD INDEXING:
 *    int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *
 * 3. MEMORY WORKFLOW:
 *    malloc → cudaMalloc → cudaMemcpy(H2D) → kernel → cudaMemcpy(D2H)
 *
 * 4. ERROR CHECKING:
 *    Always check return values of CUDA API calls!
 *
 * 5. SYNCHRONIZATION:
 *    cudaDeviceSynchronize() - wait for GPU to finish
 *
 * 6. CLEANUP:
 *    Always free both host and device memory
 *
 * ═══════════════════════════════════════════════════════════════════
 *                            EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Modify vectorAdd to perform subtraction instead
 * 2. Add a kernel for vector multiplication
 * 3. Experiment with different block sizes (64, 128, 512, 1024)
 * 4. Try different vector sizes and measure performance
 * 5. Implement error checking for all CUDA calls
 *
 * ═══════════════════════════════════════════════════════════════════
 */

