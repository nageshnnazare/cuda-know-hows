/*
 * CUDA Tutorial - Part 13: Sorting Algorithms
 * 
 * This file demonstrates parallel sorting algorithms:
 * 1. Bitonic Sort (fully parallel)
 * 2. Radix Sort (digit-by-digit)
 * 3. Merge Sort (divide-and-conquer)
 * 4. Odd-Even Sort (simple parallel)
 * 5. Quick Sort (partition-based)
 *
 * Each algorithm includes:
 * - Detailed visualization of the sorting process
 * - Step-by-step explanation
 * - Complexity analysis
 * - Performance comparison
 *
 * Compile: nvcc -o sorting 13_sorting_algorithms.cu -O3
 * Run:     ./sorting
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
 *                      BITONIC SORT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Bitonic sort is a comparison-based sorting algorithm that can be
 * efficiently parallelized. It works by recursively constructing
 * bitonic sequences and then merging them.
 *
 * Bitonic Sequence:
 * ────────────────
 * A sequence that first increases then decreases (or vice versa).
 * Examples:
 *   [1, 3, 5, 7, 6, 4, 2]  ← Bitonic (increases then decreases)
 *   [2, 4, 6, 5, 3, 1]     ← Bitonic
 *
 * Visual Example (8 elements):
 * ───────────────────────────
 * Initial:  [5, 2, 8, 1, 9, 3, 7, 4]
 *
 * Step 1: Compare-exchange pairs (distance 1)
 *          ↓↑ ↓↑ ↓↑ ↓↑
 *         [2, 5, 1, 8, 3, 9, 4, 7]
 *
 * Step 2: Compare-exchange pairs (distance 2)
 *          ↓↑   ↓↑
 *         [1, 2, 5, 8, 3, 4, 7, 9]
 *
 * Step 3: Bitonic merge
 *          ↓↑ ↓↑ ↓↑ ↓↑
 *         [1, 2, 3, 4, 5, 7, 8, 9]
 *
 * Network Diagram (8 elements):
 * ─────────────────────────────
 * [0]──────X──────────X──────────X───[0]
 *          │          │          │
 * [1]──────X──────X───│──────X───│───[1]
 *                  │   │      │   │
 * [2]──────X───────│──X───────│──X───[2]
 *          │       │          │
 * [3]──────X───────X──────────X──────[3]
 *
 * [4]──────X──────────X──────────X───[4]
 *          │          │          │
 * [5]──────X──────X───│──────X───│───[5]
 *                  │   │      │   │
 * [6]──────X───────│──X───────│──X───[6]
 *          │       │          │
 * [7]──────X───────X──────────X──────[7]
 *
 * Complexity:
 * ──────────
 * Time: O(log²n) parallel steps
 * Space: O(1) auxiliary
 * Work: O(n log²n) total comparisons
 */

// Compare and exchange elements for bitonic sort
__device__ void compareAndSwap(int *arr, int i, int j, int dir) {
    if (dir == (arr[i] > arr[j])) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// Bitonic merge kernel
__global__ void bitonicMerge(int *arr, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;  // XOR to get pair index
    
    if (ixj > idx && idx < n) {
        // Determine sort direction based on k
        // If (idx & k) == 0, sort ascending; otherwise descending
        int dir = (idx & k) == 0;
        
        if (dir == (arr[idx] > arr[ixj])) {
            // Swap
            int temp = arr[idx];
            arr[idx] = arr[ixj];
            arr[ixj] = temp;
        }
    }
}

// Shared memory bitonic sort for small arrays
__global__ void bitonicSortShared(int *arr, int n) {
    extern __shared__ int shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load into shared memory
    if (idx < n) {
        shared[tid] = arr[idx];
    } else {
        shared[tid] = INT_MAX;  // Padding with max value
    }
    __syncthreads();
    
    // Bitonic sort in shared memory
    int blockSize = blockDim.x;
    
    for (int k = 2; k <= blockSize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            
            if (ixj > tid) {
                int dir = (tid & k) == 0;
                
                if (dir == (shared[tid] > shared[ixj])) {
                    int temp = shared[tid];
                    shared[tid] = shared[ixj];
                    shared[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (idx < n) {
        arr[idx] = shared[tid];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                        RADIX SORT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Radix sort is a non-comparison sorting algorithm that processes
 * digits/bits one at a time from least significant to most significant.
 *
 * Algorithm (for decimal numbers):
 * ────────────────────────────────
 * 1. Sort by least significant digit (ones place)
 * 2. Sort by tens place (maintaining stability)
 * 3. Sort by hundreds place
 * ... continue for all digits
 *
 * Visual Example (3-digit numbers):
 * ────────────────────────────────
 * Initial:  [170, 045, 075, 090, 002, 024, 802, 066]
 *
 * Pass 1 (ones): Sort by last digit
 * ─────────────
 *   170  →  0    [170, 090, 002, 802, 024, 045, 075, 066]
 *   045  →  5
 *   075  →  5
 *   090  →  0
 *   002  →  2
 *   024  →  4
 *   802  →  2
 *   066  →  6
 *
 * Pass 2 (tens): Sort by middle digit
 * ────────────────────────────────────
 *   [002, 802, 024, 045, 066, 070, 075, 090]
 *
 * Pass 3 (hundreds): Sort by first digit
 * ──────────────────────────────────────
 *   [002, 024, 045, 066, 075, 090, 170, 802]  ← Sorted!
 *
 * For binary (typical in GPU implementations):
 * ───────────────────────────────────────────
 * Process one bit at a time (32 passes for 32-bit int)
 *
 * Example: Sorting by bit 2 (value = 4)
 * ─────────────────────────────────────
 * [5, 2, 7, 1, 6, 3]
 * Binary:
 * 5 = 101  (bit 2 = 1)  ─┐
 * 2 = 010  (bit 2 = 0)   │  Bit 2 = 0: [2, 1, 3]
 * 7 = 111  (bit 2 = 1)  ─┤
 * 1 = 001  (bit 2 = 0)   │  Bit 2 = 1: [5, 7, 6]
 * 6 = 110  (bit 2 = 1)  ─┘
 * 3 = 011  (bit 2 = 0)
 *
 * Result: [2, 1, 3, 5, 7, 6] (stable partition)
 *
 * Complexity:
 * ──────────
 * Time: O(d·n) where d is number of digits/bits
 * Space: O(n) auxiliary
 * Work: O(d·n) total
 */

// Count the number of 0s and 1s for a specific bit
__global__ void radixSortCount(int *input, int *zeros, int n, int bit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = input[idx];
        int bitValue = (value >> bit) & 1;
        
        if (bitValue == 0) {
            atomicAdd(zeros, 1);
        }
    }
}

// Scatter elements based on bit value
__global__ void radixSortScatter(int *input, int *output, int *positions,
                                 int n, int bit, int numZeros) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = input[idx];
        int bitValue = (value >> bit) & 1;
        
        if (bitValue == 0) {
            int pos = atomicAdd(&positions[0], 1);
            output[pos] = value;
        } else {
            int pos = atomicAdd(&positions[1], 1);
            output[numZeros + pos] = value;
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         MERGE SORT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Merge sort is a divide-and-conquer algorithm that recursively splits
 * the array and merges sorted subarrays.
 *
 * Algorithm:
 * ─────────
 * 1. Divide array into two halves
 * 2. Recursively sort each half
 * 3. Merge the two sorted halves
 *
 * Visual Example:
 * ──────────────
 *               [5, 2, 8, 1, 9, 3, 7, 4]
 *                 /                \
 *         [5, 2, 8, 1]        [9, 3, 7, 4]
 *           /      \              /      \
 *       [5, 2]  [8, 1]        [9, 3]  [7, 4]
 *        /  \    /  \          /  \    /  \
 *      [5] [2] [8] [1]       [9] [3] [7] [4]
 *        \  /    \  /          \  /    \  /
 *       [2, 5]  [1, 8]        [3, 9]  [4, 7]
 *           \      /              \      /
 *         [1, 2, 5, 8]        [3, 4, 7, 9]
 *                 \                /
 *               [1, 2, 3, 4, 5, 7, 8, 9]
 *
 * Merge Operation:
 * ───────────────
 * Merge [2, 5, 8] and [1, 3, 9]:
 *
 *   Array 1: [2, 5, 8]     Array 2: [1, 3, 9]
 *             ↑                       ↑
 *   Compare: 2 vs 1 → pick 1     Result: [1]
 *
 *   Array 1: [2, 5, 8]     Array 2: [1, 3, 9]
 *             ↑                          ↑
 *   Compare: 2 vs 3 → pick 2     Result: [1, 2]
 *
 *   ... continue until Result: [1, 2, 3, 5, 8, 9]
 *
 * GPU Parallelization:
 * ───────────────────
 * - Each thread merges a small chunk
 * - Progressively merge larger chunks
 * - Bottom-up approach works well
 *
 * Complexity:
 * ──────────
 * Time: O(log n) parallel steps
 * Space: O(n) auxiliary
 * Work: O(n log n) total
 */

// Merge two sorted subarrays
__global__ void mergeKernel(int *input, int *output, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = 2 * idx * width;
    
    if (start >= n) return;
    
    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);
    
    int i = start;
    int j = mid;
    int k = start;
    
    // Merge input[start..mid-1] and input[mid..end-1]
    while (i < mid && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    
    // Copy remaining elements
    while (i < mid) {
        output[k++] = input[i++];
    }
    while (j < end) {
        output[k++] = input[j++];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                       ODD-EVEN SORT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Odd-even sort (also known as brick sort) is a simple parallel sorting
 * algorithm that alternates between comparing odd and even indexed pairs.
 *
 * Algorithm:
 * ─────────
 * Repeat until sorted:
 *   1. Odd phase:  Compare (0,1), (2,3), (4,5), ...
 *   2. Even phase: Compare (1,2), (3,4), (5,6), ...
 *
 * Visual Example:
 * ──────────────
 * Initial: [5, 2, 8, 1, 9, 3]
 *
 * Odd phase (compare pairs 0-1, 2-3, 4-5):
 *           ↓↑     ↓↑     ↓↑
 *          [2, 5,  1, 8,  3, 9]
 *
 * Even phase (compare pairs 1-2, 3-4):
 *              ↓↑     ↓↑
 *          [2, 1, 5, 3, 8, 9]
 *
 * Odd phase:
 *           ↓↑     ↓↑     ↓↑
 *          [1, 2,  3, 5,  8, 9]  ← Sorted!
 *
 * Parallel Execution:
 * ──────────────────
 * Odd Phase:        Even Phase:
 * Thread 0: [0,1]   Thread 0: [1,2]
 * Thread 1: [2,3]   Thread 1: [3,4]
 * Thread 2: [4,5]   Thread 2: [5,6]
 * (All threads run simultaneously)
 *
 * Complexity:
 * ──────────
 * Time: O(n) parallel steps (worst case)
 * Space: O(1) auxiliary
 * Work: O(n²) total comparisons
 */

__global__ void oddEvenSortOdd(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * idx;  // Start at even indices
    
    if (i + 1 < n) {
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

__global__ void oddEvenSortEven(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * idx + 1;  // Start at odd indices
    
    if (i + 1 < n) {
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 */

// Generate random array
void generateRandomArray(int *arr, int n, int maxValue) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % maxValue;
    }
}

// Verify array is sorted
bool verifySorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("ERROR: Array not sorted at index %d (%d > %d)\n", 
                   i, arr[i], arr[i + 1]);
            return false;
        }
    }
    return true;
}

// Print array (first and last few elements)
void printArray(const char *label, int *arr, int n, int showCount) {
    printf("%s: [", label);
    for (int i = 0; i < showCount && i < n; i++) {
        printf("%d", arr[i]);
        if (i < showCount - 1 && i < n - 1) printf(", ");
    }
    if (n > showCount) printf(", ...");
    if (n > showCount) {
        printf(", ");
        for (int i = n - 3; i < n; i++) {
            printf("%d", arr[i]);
            if (i < n - 1) printf(", ");
        }
    }
    printf("]\n");
}

// CPU reference sort (for verification)
void cpuSort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Tutorial: Sorting Algorithms              ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    srand(time(NULL));
    
    const int N = 1 << 20;  // 1M elements
    const int MAX_VALUE = 10000;
    
    printf("Array size: %d elements\n", N);
    printf("Value range: [0, %d]\n\n", MAX_VALUE - 1);
    
    // Allocate host memory
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));
    int *h_reference = (int*)malloc(N * sizeof(int));
    
    generateRandomArray(h_input, N, MAX_VALUE);
    memcpy(h_reference, h_input, N * sizeof(int));
    
    printArray("Input (sample)", h_input, N, 10);
    printf("\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Bitonic Sort (power of 2 size only)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Bitonic Sort\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int bitonicN = 1 << 20;  // Must be power of 2
    int *d_bitonic;
    CUDA_CHECK(cudaMalloc(&d_bitonic, bitonicN * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_bitonic, h_input, bitonicN * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (bitonicN + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Bitonic sort phases
    for (int k = 2; k <= bitonicN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicMerge<<<gridSize, blockSize>>>(d_bitonic, bitonicN, j, k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float bitonicTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&bitonicTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_bitonic, bitonicN * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Bitonic Sort:\n");
    printf("  Time: %.3f ms\n", bitonicTime);
    printf("  Throughput: %.2f M elements/sec\n", 
           bitonicN / (bitonicTime * 1000.0f));
    printf("  Verified: %s\n\n", verifySorted(h_output, bitonicN) ? "✓" : "✗");
    
    printArray("Output (sample)", h_output, bitonicN, 10);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_bitonic));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Radix Sort (bit-by-bit)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Radix Sort\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int *d_radix_in, *d_radix_out, *d_zeros, *d_positions;
    CUDA_CHECK(cudaMalloc(&d_radix_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radix_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_zeros, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_positions, 2 * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_radix_in, h_input, N * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Sort bit by bit (32 bits for int)
    for (int bit = 0; bit < 32; bit++) {
        // Count zeros
        CUDA_CHECK(cudaMemset(d_zeros, 0, sizeof(int)));
        radixSortCount<<<gridSize, blockSize>>>(d_radix_in, d_zeros, N, bit);
        
        // Scatter
        int h_zeros;
        CUDA_CHECK(cudaMemcpy(&h_zeros, d_zeros, sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        int h_positions[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(d_positions, h_positions, 2 * sizeof(int),
                             cudaMemcpyHostToDevice));
        
        radixSortScatter<<<gridSize, blockSize>>>(d_radix_in, d_radix_out,
                                                    d_positions, N, bit, h_zeros);
        
        // Swap buffers
        int *temp = d_radix_in;
        d_radix_in = d_radix_out;
        d_radix_out = temp;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float radixTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&radixTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_radix_in, N * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Radix Sort (32-bit):\n");
    printf("  Time: %.3f ms\n", radixTime);
    printf("  Throughput: %.2f M elements/sec\n", N / (radixTime * 1000.0f));
    printf("  Verified: %s\n\n", verifySorted(h_output, N) ? "✓" : "✗");
    
    printArray("Output (sample)", h_output, N, 10);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_radix_in));
    CUDA_CHECK(cudaFree(d_radix_out));
    CUDA_CHECK(cudaFree(d_zeros));
    CUDA_CHECK(cudaFree(d_positions));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Odd-Even Sort (small array for demonstration)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Odd-Even Sort (Small Array Demo)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int oddEvenN = 1024;
    int *d_oddEven;
    CUDA_CHECK(cudaMalloc(&d_oddEven, oddEvenN * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_oddEven, h_input, oddEvenN * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    int oddGridSize = (oddEvenN / 2 + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Perform n phases (worst case)
    for (int phase = 0; phase < oddEvenN; phase++) {
        if (phase % 2 == 0) {
            oddEvenSortOdd<<<oddGridSize, blockSize>>>(d_oddEven, oddEvenN);
        } else {
            oddEvenSortEven<<<oddGridSize, blockSize>>>(d_oddEven, oddEvenN);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float oddEvenTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&oddEvenTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_oddEven, oddEvenN * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Odd-Even Sort (%d elements):\n", oddEvenN);
    printf("  Time: %.3f ms\n", oddEvenTime);
    printf("  Verified: %s\n\n", verifySorted(h_output, oddEvenN) ? "✓" : "✗");
    
    CUDA_CHECK(cudaFree(d_oddEven));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Performance Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Performance Summary (1M elements)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("┌─────────────────┬──────────┬──────────────┬──────────┐\n");
    printf("│ Algorithm       │ Time(ms) │ Throughput   │ Best For │\n");
    printf("├─────────────────┼──────────┼──────────────┼──────────┤\n");
    printf("│ Bitonic Sort    │ %7.2f  │ %7.2f M/s│ Power-2  │\n", 
           bitonicTime, N / (bitonicTime * 1000.0f));
    printf("│ Radix Sort      │ %7.2f  │ %7.2f M/s│ Integers │\n", 
           radixTime, N / (radixTime * 1000.0f));
    printf("│ Odd-Even Sort*  │ %7.2f  │    N/A       │ Simple   │\n", 
           oddEvenTime);
    printf("└─────────────────┴──────────┴──────────────┴──────────┘\n");
    printf("* Tested on 1K elements only\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    free(h_input);
    free(h_output);
    free(h_reference);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Bitonic sort: Great for power-of-2 sizes          ║\n");
    printf("║ 2. Radix sort: Linear time for integers              ║\n");
    printf("║ 3. Merge sort: Divide-and-conquer parallelizes well  ║\n");
    printf("║ 4. Choose algorithm based on data characteristics    ║\n");
    printf("║ 5. Use Thrust library for production code            ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement full merge sort with all levels
 * 2. Add sample sort (bucketbased)
 * 3. Implement counting sort for small ranges
 * 4. Add key-value pair sorting
 * 5. Optimize radix sort with shared memory histograms
 * 6. Implement stable sorting verification
 * 7. Add multi-GPU sorting for very large datasets
 * 8. Compare with Thrust library performance
 *
 * ═══════════════════════════════════════════════════════════════════
 */

