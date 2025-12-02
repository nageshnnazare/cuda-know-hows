/*
 * CUDA Tutorial - Part 5: Matrix Operations
 * 
 * This file demonstrates:
 * 1. Matrix multiplication (naive and optimized)
 * 2. Matrix transpose
 * 3. Tiling techniques
 * 4. Performance optimization strategies
 * 5. Comparison with cuBLAS
 *
 * Compile: nvcc -o matrix_ops 05_matrix_operations.cu -lcublas
 * Run:     ./matrix_ops
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    MATRIX MULTIPLICATION OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Matrix Multiplication: C = A × B
 * 
 * Where:
 *   A is M×K
 *   B is K×N
 *   C is M×N
 *
 * Formula: C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
 *
 * Visual Example (3×3 matrices):
 * ═══════════════════════════════
 *
 *         B                      
 *     ┌───────────┐               
 *     │b00 b01 b02│              
 *     │b10 b11 b12│              
 *     │b20 b21 b22│              
 *     └───────────┘               
 *
 * A ┌───────────┐    C
 *   │a00 a01 a02│  ┌───────────┐
 *   │a10 a11 a12│  │c00 c01 c02│
 *   │a20 a21 a22│  │c10 c11 c12│
 *   └───────────┘  │c20 c21 c22│
 *                  └───────────┘
 *
 * Computing C[1,2] (one element):
 * ───────────────────────────────
 *   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
 *          = a10*b02 + a11*b12 + a12*b22
 *
 * Memory Layout (Row-Major):
 * ──────────────────────────
 *   A = [a00,a01,a02, a10,a11,a12, a20,a21,a22]
 *   Index for A[i,j] = i * width + j
 */

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    NAIVE MATRIX MULTIPLICATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Algorithm:
 * - Each thread computes one element of C
 * - Reads entire row from A and column from B
 * - No optimization, many global memory accesses
 *
 * Thread Organization:
 * ────────────────────
 *   Thread(i,j) computes C[i,j]
 *
 *   Grid of Blocks
 *   ┌──┬──┬──┬──┐
 *   │  │  │  │  │
 *   ├──┼──┼──┼──┤
 *   │  │  │  │  │  Each block computes
 *   ├──┼──┼──┼──┤  a tile of C
 *   │  │  │  │  │
 *   └──┴──┴──┴──┘
 *
 * Memory Access Pattern:
 * ──────────────────────
 *   For C[i,j]:
 *     Read A[i,0], A[i,1], ..., A[i,K-1]  (row)
 *     Read B[0,j], B[1,j], ..., B[K-1,j]  (column)
 *
 *   Problem: Each element of A and B is read multiple times!
 *   If N=M=K=1024, each element is read 1024 times.
 */

__global__ void matrixMulNaive(float *A, float *B, float *C, 
                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                 TILED MATRIX MULTIPLICATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Optimization: Use shared memory to cache tiles
 * 
 * Strategy:
 * 1. Divide matrices into tiles
 * 2. Load tile into shared memory (fast)
 * 3. Compute partial products using shared memory
 * 4. Repeat for all tiles
 *
 * Visualization (Tiling Process):
 * ════════════════════════════════
 *
 * Step 1: Load first tiles
 *         B
 *     ┌───┬───┬───┐
 *     │T0 │   │   │
 *     ├───┼───┼───┤
 *     │   │   │   │
 * A   └───┴───┴───┘
 * ┌───┬───┬───┐       C
 * │T0 │   │   │   ┌───┬───┬───┐
 * ├───┼───┼───┤   │ ? │   │   │
 * │   │   │   │   ├───┼───┼───┤
 * └───┴───┴───┘   │   │   │   │
 *                 └───┴───┴───┘
 *
 * Compute partial product of loaded tiles
 *
 * Step 2: Load next tiles
 *         B
 *     ┌───┬───┬───┐
 *     │   │T1 │   │
 *     ├───┼───┼───┤
 *     │   │   │   │
 * A   └───┴───┴───┘
 * ┌───┬───┬───┐       C
 * │   │T1 │   │   ┌───┬───┬───┐
 * ├───┼───┼───┤   │ + │   │   │
 * │   │   │   │   ├───┼───┼───┤
 * └───┴───┴───┘   │   │   │   │
 *                 └───┴───┴───┘
 *
 * Add to previous result, repeat...
 *
 * Benefits:
 * - Reduces global memory accesses
 * - Uses fast shared memory
 * - Improves cache locality
 */

__global__ void matrixMulTiled(float *A, float *B, float *C,
                               int M, int K, int N) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      MATRIX TRANSPOSE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Transpose: A^T[i,j] = A[j,i]
 *
 * Example:
 * ────────
 *   A = │1 2 3│     A^T = │1 4│
 *       │4 5 6│           │2 5│
 *                         │3 6│
 *
 * Challenge: Avoiding bank conflicts in shared memory
 *
 * Naive Transpose (has bank conflicts):
 * ─────────────────────────────────────
 *   Read from A: coalesced
 *   Write to A^T: strided (bad!)
 *
 * Optimized: Use shared memory with padding
 */

// Naive transpose
__global__ void transposeNaive(float *input, float *output, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Optimized transpose using shared memory
__global__ void transposeOptimized(float *input, float *output,
                                   int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Transpose coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write tile (coalesced in output)
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                      UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 */

void initMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

void printMatrix(float *mat, int rows, int cols, const char *name) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    int maxRows = (rows > 8) ? 8 : rows;
    int maxCols = (cols > 8) ? 8 : cols;
    
    for (int i = 0; i < maxRows; i++) {
        printf("  ");
        for (int j = 0; j < maxCols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("  ...\n");
}

bool verifyMatrixMul(float *A, float *B, float *C, int M, int K, int N) {
    const float epsilon = 1e-3;
    
    for (int i = 0; i < M && i < 100; i++) {  // Check first 100 rows
        for (int j = 0; j < N && j < 100; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            
            if (fabsf(C[i * N + j] - sum) > epsilon) {
                printf("Mismatch at [%d,%d]: expected %.5f, got %.5f\n",
                       i, j, sum, C[i * N + j]);
                return false;
            }
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
    printf("║         CUDA Tutorial: Matrix Operations              ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Matrix dimensions
    int M = 1024;  // Rows of A and C
    int K = 512;   // Cols of A, Rows of B
    int N = 1024;  // Cols of B and C
    
    printf("Matrix dimensions:\n");
    printf("  A: %d × %d\n", M, K);
    printf("  B: %d × %d\n", K, N);
    printf("  C: %d × %d\n\n", M, N);
    
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    
    printf("Memory required:\n");
    printf("  A: %.2f MB\n", bytesA / (1024.0f * 1024.0f));
    printf("  B: %.2f MB\n", bytesB / (1024.0f * 1024.0f));
    printf("  C: %.2f MB\n", bytesC / (1024.0f * 1024.0f));
    printf("  Total: %.2f MB\n\n", 
           (bytesA + bytesB + bytesC) / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    
    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    
    // Setup execution
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Naive Matrix Multiplication
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Naive Matrix Multiplication\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float naiveTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = verifyMatrixMul(h_A, h_B, h_C, M, K, N);
    
    // Calculate GFLOPS
    float numOps = 2.0f * M * N * K;  // Multiply-add counts as 2 ops
    float gflops = (numOps / 1e9f) / (naiveTime / 1000.0f);
    
    printf("Naive implementation:\n");
    printf("  Time: %.3f ms\n", naiveTime);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Verification: %s\n\n", correct ? "✓ PASSED" : "✗ FAILED");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Tiled Matrix Multiplication
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Tiled Matrix Multiplication\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float tiledTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    correct = verifyMatrixMul(h_A, h_B, h_C, M, K, N);
    
    float tiledGflops = (numOps / 1e9f) / (tiledTime / 1000.0f);
    
    printf("Tiled implementation (tile size=%d):\n", TILE_SIZE);
    printf("  Time: %.3f ms\n", tiledTime);
    printf("  Performance: %.2f GFLOPS\n", tiledGflops);
    printf("  Speedup over naive: %.2fx\n", naiveTime / tiledTime);
    printf("  Verification: %s\n\n", correct ? "✓ PASSED" : "✗ FAILED");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: cuBLAS (Optimized Library)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: cuBLAS (Highly Optimized)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CUDA_CHECK(cudaEventRecord(start));
    // cuBLAS uses column-major, so we compute B*A to get C in row-major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float cublasTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cublasTime, start, stop));
    
    float cublasGflops = (numOps / 1e9f) / (cublasTime / 1000.0f);
    
    printf("cuBLAS (NVIDIA optimized):\n");
    printf("  Time: %.3f ms\n", cublasTime);
    printf("  Performance: %.2f GFLOPS\n", cublasGflops);
    printf("  Speedup over naive: %.2fx\n", naiveTime / cublasTime);
    printf("  Speedup over tiled: %.2fx\n\n", tiledTime / cublasTime);
    
    cublasDestroy(handle);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Performance Comparison Table
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Performance Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("┌─────────────────┬────────────┬────────────┬──────────┐\n");
    printf("│ Implementation  │ Time (ms)  │ GFLOPS     │ Speedup  │\n");
    printf("├─────────────────┼────────────┼────────────┼──────────┤\n");
    printf("│ Naive           │ %9.3f  │ %9.2f  │ 1.00x    │\n", 
           naiveTime, gflops);
    printf("│ Tiled           │ %9.3f  │ %9.2f  │ %.2fx    │\n", 
           tiledTime, tiledGflops, naiveTime / tiledTime);
    printf("│ cuBLAS          │ %9.3f  │ %9.2f  │ %.2fx    │\n", 
           cublasTime, cublasGflops, naiveTime / cublasTime);
    printf("└─────────────────┴────────────┴────────────┴──────────┘\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 4: Matrix Transpose
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 4: Matrix Transpose\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int transposeSize = 2048;
    size_t transposeBytes = transposeSize * transposeSize * sizeof(float);
    
    float *h_trans = (float*)malloc(transposeBytes);
    float *d_trans, *d_transOut;
    CUDA_CHECK(cudaMalloc(&d_trans, transposeBytes));
    CUDA_CHECK(cudaMalloc(&d_transOut, transposeBytes));
    
    initMatrix(h_trans, transposeSize, transposeSize);
    CUDA_CHECK(cudaMemcpy(d_trans, h_trans, transposeBytes, cudaMemcpyHostToDevice));
    
    dim3 transBlock(TILE_SIZE, TILE_SIZE);
    dim3 transGrid((transposeSize + TILE_SIZE - 1) / TILE_SIZE,
                   (transposeSize + TILE_SIZE - 1) / TILE_SIZE);
    
    // Naive transpose
    CUDA_CHECK(cudaEventRecord(start));
    transposeNaive<<<transGrid, transBlock>>>(d_trans, d_transOut, 
                                              transposeSize, transposeSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float naiveTransTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTransTime, start, stop));
    
    // Optimized transpose
    CUDA_CHECK(cudaEventRecord(start));
    transposeOptimized<<<transGrid, transBlock>>>(d_trans, d_transOut,
                                                  transposeSize, transposeSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float optTransTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&optTransTime, start, stop));
    
    float bandwidth = (2.0f * transposeBytes) / (1024.0f * 1024.0f * 1024.0f) / 
                      (optTransTime / 1000.0f);
    
    printf("Matrix: %dx%d (%.2f MB)\n\n", transposeSize, transposeSize,
           transposeBytes / (1024.0f * 1024.0f));
    
    printf("┌─────────────────┬────────────┬──────────────┐\n");
    printf("│ Implementation  │ Time (ms)  │ Bandwidth    │\n");
    printf("├─────────────────┼────────────┼──────────────┤\n");
    printf("│ Naive           │ %9.3f  │ %8.2f GB/s │\n", 
           naiveTransTime, 
           (2.0f * transposeBytes) / (1024.0f * 1024.0f * 1024.0f) / 
           (naiveTransTime / 1000.0f));
    printf("│ Optimized       │ %9.3f  │ %8.2f GB/s │\n", 
           optTransTime, bandwidth);
    printf("└─────────────────┴────────────┴──────────────┘\n\n");
    
    printf("Speedup: %.2fx\n\n", naiveTransTime / optTransTime);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_trans));
    CUDA_CHECK(cudaFree(d_transOut));
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_trans);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Tiling dramatically improves performance           ║\n");
    printf("║ 2. Shared memory reduces global memory traffic       ║\n");
    printf("║ 3. cuBLAS is highly optimized - use it when possible ║\n");
    printf("║ 4. Memory access patterns matter (coalescing)        ║\n");
    printf("║ 5. Transpose benefits from shared memory padding     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement matrix-vector multiplication
 * 2. Add support for non-square matrices in all kernels
 * 3. Implement strided matrix multiplication
 * 4. Optimize for different tile sizes (8, 16, 32)
 * 5. Implement matrix addition and element-wise operations
 * 6. Add batch matrix multiplication support
 *
 * ═══════════════════════════════════════════════════════════════════
 */

