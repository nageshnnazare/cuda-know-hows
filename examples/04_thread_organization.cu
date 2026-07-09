/*
 * CUDA Tutorial - Part 4: Thread Organization and Indexing
 * 
 * This file demonstrates:
 * 1. Understanding thread hierarchy (Grid → Block → Thread)
 * 2. 1D, 2D, and 3D thread organization
 * 3. Thread indexing in different dimensions
 * 4. Choosing optimal block and grid sizes
 * 5. Practical examples with images and matrices
 *
 * Compile: nvcc -o thread_org 04_thread_organization.cu
 * Run:     ./thread_org
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
 *                   THREAD HIERARCHY VISUALIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * CUDA organizes threads in a 3-level hierarchy:
 *
 *   Grid (all threads in kernel launch)
 *     │
 *     └─> Block (group of threads that can cooperate)
 *           │
 *           └─> Thread (individual execution unit)
 *
 * Each level can be 1D, 2D, or 3D:
 *
 * 1D Configuration:
 * ═════════════════
 *   Grid(4, 1, 1)
 *   ┌────┬────┬────┬────┐
 *   │ B0 │ B1 │ B2 │ B3 │  Each block has...
 *   └────┴────┴────┴────┘
 *
 *   Block(8, 1, 1)
 *   ┌──┬──┬──┬──┬──┬──┬──┬──┐
 *   │T0│T1│T2│T3│T4│T5│T6│T7│
 *   └──┴──┴──┴──┴──┴──┴──┴──┘
 *
 * 2D Configuration (Common for image processing):
 * ═════════════════════════════════════════════
 *   Grid(2, 2, 1)          Block(4, 4, 1)
 *   ┌────┬────┐            ┌──┬──┬──┬──┐
 *   │ B0 │ B1 │            │00│01│02│03│
 *   ├────┼────┤            ├──┼──┼──┼──┤
 *   │ B2 │ B3 │            │10│11│12│13│
 *   └────┴────┘            ├──┼──┼──┼──┤
 *                          │20│21│22│23│
 *                          ├──┼──┼──┼──┤
 *                          │30│31│32│33│
 *                          └──┴──┴──┴──┘
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      1D THREAD ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════
 */

// Simple 1D kernel that prints thread information
__global__ void print1DInfo(int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int gridSize = gridDim.x;
    int globalId = bid * blockSize + tid;
    
    if (globalId < 20) {  // Only print first 20 threads
        printf("Thread[%2d]: Block=%d, LocalID=%d, GlobalID=%d, "
               "GridSize=%d, BlockSize=%d\n",
               globalId, bid, tid, globalId, gridSize, blockSize);
    }
}

// 1D vector processing kernel
__global__ void process1D(float *data, int n) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalId < n) {
        data[globalId] = sqrt(data[globalId] * data[globalId] + 1.0f);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      2D THREAD ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * For 2D data (images, matrices), use 2D thread organization:
 *
 * Global Index Calculation:
 * ────────────────────────
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 *   index = row * width + col
 *
 * Example: 8x8 image with 4x4 blocks
 * ───────────────────────────────────
 *
 *        Blocks (2x2 grid)
 *        ┌────────┬────────┐
 *        │(0,1)   │(1,1)   │
 *        │B0      │B1      │
 *        ├────────┼────────┤
 *        │(0,0)   │(1,0)   │
 *        │B2      │B3      │
 *        └────────┴────────┘
 *
 *        Block(0,0) = B2
 *        ┌──┬──┬──┬──┐
 *        │03│13│23│33│
 *        ├──┼──┼──┼──┤
 *        │02│12│22│32│
 *        ├──┼──┼──┼──┤
 *        │01│11│21│31│
 *        ├──┼──┼──┼──┤
 *        │00│10│20│30│
 *        └──┴──┴──┴──┘
 *        (threadIdx.x, threadIdx.y)
 */

// Print 2D thread information
__global__ void print2DInfo(int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;
    
    if (row < 4 && col < 4) {  // Only print 4x4 region
        printf("Thread[%2d,%2d]: Block(%d,%d) Local(%d,%d) Global(%d,%d) Idx=%d\n",
               col, row, blockIdx.x, blockIdx.y, 
               threadIdx.x, threadIdx.y, col, row, idx);
    }
}

// 2D image processing: convert to grayscale
__global__ void rgbToGrayscale(unsigned char *rgb, unsigned char *gray, 
                               int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int grayIdx = row * width + col;
        int rgbIdx = grayIdx * 3;
        
        // Grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        
        gray[grayIdx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// 2D matrix addition
__global__ void matrixAdd2D(float *A, float *B, float *C, 
                            int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      3D THREAD ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * For 3D data (volumes, video, 3D simulations):
 *
 * Global Index Calculation:
 * ────────────────────────
 *   x = blockIdx.x * blockDim.x + threadIdx.x
 *   y = blockIdx.y * blockDim.y + threadIdx.y
 *   z = blockIdx.z * blockDim.z + threadIdx.z
 *   index = z * (width * height) + y * width + x
 *
 * Visualization (4x4x4 volume):
 * ─────────────────────────────
 *
 *     Z-axis (depth)
 *     ↑
 *     │    Layer 0    Layer 1    Layer 2    Layer 3
 *     │   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
 *     │   │ ░░░ │    │ ░░░ │    │ ░░░ │    │ ░░░ │
 *     │   │ ░░░ │    │ ░░░ │    │ ░░░ │    │ ░░░ │
 *     │   └─────┘    └─────┘    └─────┘    └─────┘
 *     └──────────────────────────────────────────────> X-axis (width)
 *        /
 *       /
 *      ↙ Y-axis (height)
 */

// 3D volume processing
__global__ void process3D(float *volume, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = z * (width * height) + y * width + x;
        
        // Example: Gaussian-like function
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float cz = depth / 2.0f;
        
        float dx = (x - cx) / cx;
        float dy = (y - cy) / cy;
        float dz = (z - cz) / cz;
        
        volume[idx] = expf(-(dx*dx + dy*dy + dz*dz));
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                 CHOOSING BLOCK AND GRID SIZES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Guidelines:
 * ──────────
 * 1. Block size should be multiple of warp size (32)
 * 2. Common block sizes: 128, 256, 512, 1024
 * 3. For 2D: 16x16=256, 32x32=1024 are popular
 * 4. For 3D: 8x8x8=512, 16x16x4=1024 work well
 * 5. Grid size: (N + blockSize - 1) / blockSize
 *
 * Occupancy Considerations:
 * ────────────────────────
 * - More threads per block → better latency hiding
 * - But limited by: registers, shared memory, max threads
 * - Use CUDA Occupancy Calculator for optimization
 */

void printBlockGridInfo(dim3 grid, dim3 block, const char *name) {
    printf("\n%s Configuration:\n", name);
    printf("  Grid:  (%d, %d, %d) = %d blocks\n", 
           grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);
    printf("  Block: (%d, %d, %d) = %d threads/block\n", 
           block.x, block.y, block.z, block.x * block.y * block.z);
    printf("  Total: %d threads\n", 
           grid.x * grid.y * grid.z * block.x * block.y * block.z);
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      PERFORMANCE TESTING
 * ═══════════════════════════════════════════════════════════════════
 */

__global__ void matrixMulSimple(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float testBlockSize(int N, dim3 blockDim) {
    size_t bytes = N * N * sizeof(float);
    
    // Allocate memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Initialize (simple pattern)
    CUDA_CHECK(cudaMemset(d_A, 1, bytes));
    CUDA_CHECK(cudaMemset(d_B, 1, bytes));
    
    // Calculate grid size
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    
    // Time the kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulSimple<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║      CUDA Tutorial: Thread Organization & Indexing    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("📊 Device: %s\n", prop.name);
    printf("   Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("   Max block dimensions: [%d, %d, %d]\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("   Max grid dimensions: [%d, %d, %d]\n\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 1: 1D Thread Organization
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 1: 1D Thread Organization\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    int n1d = 100;
    dim3 block1d(32);
    dim3 grid1d((n1d + block1d.x - 1) / block1d.x);
    
    printBlockGridInfo(grid1d, block1d, "1D");
    
    printf("\nThread Information (first 20 threads):\n");
    printf("──────────────────────────────────────────────────────\n");
    print1DInfo<<<grid1d, block1d>>>(n1d);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 2: 2D Thread Organization
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Example 2: 2D Thread Organization\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    int width = 16, height = 16;
    dim3 block2d(4, 4);  // 16 threads per block
    dim3 grid2d((width + block2d.x - 1) / block2d.x,
                (height + block2d.y - 1) / block2d.y);
    
    printBlockGridInfo(grid2d, block2d, "2D");
    
    printf("\nThread Information (4x4 region):\n");
    printf("──────────────────────────────────────────────────────\n");
    print2DInfo<<<grid2d, block2d>>>(width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Test 2D image processing
    printf("\n2D Image Processing Example:\n");
    printf("──────────────────────────────────────────────────────\n");
    
    int imgWidth = 1920, imgHeight = 1080;
    size_t rgbBytes = imgWidth * imgHeight * 3 * sizeof(unsigned char);
    size_t grayBytes = imgWidth * imgHeight * sizeof(unsigned char);
    
    unsigned char *d_rgb, *d_gray;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
    
    // Use 16x16 blocks for image processing (common choice)
    dim3 blockImg(16, 16);
    dim3 gridImg((imgWidth + blockImg.x - 1) / blockImg.x,
                 (imgHeight + blockImg.y - 1) / blockImg.y);
    
    printf("Image: %dx%d pixels\n", imgWidth, imgHeight);
    printf("Block: %dx%d = %d threads\n", blockImg.x, blockImg.y, 
           blockImg.x * blockImg.y);
    printf("Grid:  %dx%d = %d blocks\n", gridImg.x, gridImg.y, 
           gridImg.x * gridImg.y);
    printf("Total: %d threads\n", 
           gridImg.x * gridImg.y * blockImg.x * blockImg.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    rgbToGrayscale<<<gridImg, blockImg>>>(d_rgb, d_gray, imgWidth, imgHeight);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float imgTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&imgTime, start, stop));
    printf("Conversion time: %.3f ms\n", imgTime);
    printf("Throughput: %.2f Gpixels/s\n", 
           (imgWidth * imgHeight) / (imgTime * 1e6));
    
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 3: 3D Thread Organization
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Example 3: 3D Thread Organization\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    int volWidth = 256, volHeight = 256, volDepth = 128;
    size_t volBytes = volWidth * volHeight * volDepth * sizeof(float);
    
    float *d_volume;
    CUDA_CHECK(cudaMalloc(&d_volume, volBytes));
    
    // 8x8x8 blocks are common for 3D (512 threads)
    dim3 block3d(8, 8, 8);
    dim3 grid3d((volWidth + block3d.x - 1) / block3d.x,
                (volHeight + block3d.y - 1) / block3d.y,
                (volDepth + block3d.z - 1) / block3d.z);
    
    printBlockGridInfo(grid3d, block3d, "3D");
    
    printf("\nProcessing %dx%dx%d volume (%.2f MB)\n", 
           volWidth, volHeight, volDepth, volBytes / (1024.0f * 1024.0f));
    
    CUDA_CHECK(cudaEventRecord(start));
    process3D<<<grid3d, block3d>>>(d_volume, volWidth, volHeight, volDepth);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float volTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&volTime, start, stop));
    printf("Processing time: %.3f ms\n", volTime);
    printf("Throughput: %.2f GB/s\n", 
           volBytes / (volTime / 1000.0f) / (1024.0f * 1024.0f * 1024.0f));
    
    CUDA_CHECK(cudaFree(d_volume));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 4: Block Size Impact on Performance
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Example 4: Block Size Impact on Performance\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    int matSize = 512;
    printf("\nMatrix multiplication: %dx%d\n", matSize, matSize);
    printf("Testing different block sizes...\n\n");
    
    printf("┌──────────────┬──────────────┬───────────────┐\n");
    printf("│ Block Size   │ Time (ms)    │ Rel. Perf.    │\n");
    printf("├──────────────┼──────────────┼───────────────┤\n");
    
    struct { int x; int y; } blockSizes[] = {
        {8, 8}, {16, 16}, {32, 32}
    };
    
    float baseTime = 0;
    for (int i = 0; i < 3; i++) {
        dim3 blockTest(blockSizes[i].x, blockSizes[i].y);
        float time = testBlockSize(matSize, blockTest);
        
        if (i == 0) baseTime = time;
        
        printf("│ %2dx%-2d=%4d   │ %10.3f   │ %10.2fx    │\n",
               blockSizes[i].x, blockSizes[i].y, 
               blockSizes[i].x * blockSizes[i].y,
               time, baseTime / time);
    }
    printf("└──────────────┴──────────────┴───────────────┘\n");
    
    printf("\n💡 Observation:\n");
    printf("   Block size affects performance due to:\n");
    printf("   - Occupancy (threads available to hide latency)\n");
    printf("   - Resource usage (registers, shared memory)\n");
    printf("   - Warp scheduling efficiency\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Visual Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Thread Organization Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("  DIMENSION │ USE CASE         │ TYPICAL BLOCKS  │\n");
    printf("  ──────────┼──────────────────┼─────────────────┤\n");
    printf("  1D        │ Vectors, arrays  │ 128, 256, 512   │\n");
    printf("  2D        │ Images, matrices │ 16x16, 32x32    │\n");
    printf("  3D        │ Volumes, cubes   │ 8x8x8, 16x16x4  │\n\n");
    
    printf("  INDEX FORMULAS:\n");
    printf("  ──────────────────────────────────────────────────\n");
    printf("  1D: idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("\n");
    printf("  2D: row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("      col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("      idx = row * width + col\n");
    printf("\n");
    printf("  3D: x = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("      y = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("      z = blockIdx.z * blockDim.z + threadIdx.z\n");
    printf("      idx = z * (width*height) + y * width + x\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Match thread organization to data dimensionality  ║\n");
    printf("║ 2. Block size should be multiple of warp size (32)   ║\n");
    printf("║ 3. Common 2D blocks: 16x16 (256) or 32x32 (1024)     ║\n");
    printf("║ 4. Always check boundaries (if idx < n)              ║\n");
    printf("║ 5. Experiment with block sizes for best performance  ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement a 2D Gaussian blur kernel
 * 2. Create a 3D heat diffusion simulation
 * 3. Test block sizes from 64 to 1024 and plot performance
 * 4. Implement image rotation using 2D threads
 * 5. Write a kernel that processes a 4D tensor (batch, depth, height, width)
 *
 * ═══════════════════════════════════════════════════════════════════
 */

