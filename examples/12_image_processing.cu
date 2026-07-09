/*
 * CUDA Tutorial - Part 12: Image Processing
 * 
 * This file demonstrates practical image processing algorithms:
 * 1. Convolution with various kernels (Gaussian blur, sharpen)
 * 2. Sobel edge detection
 * 3. Box filter (simple blur)
 * 4. Histogram computation and equalization
 * 5. Bilateral filter (edge-preserving)
 * 6. Median filter (noise reduction)
 *
 * Each algorithm includes:
 * - Detailed explanation with ASCII art
 * - Multiple optimization levels
 * - Performance measurements
 *
 * Compile: nvcc -o image_proc 12_image_processing.cu -O3
 * Run:     ./image_proc
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
 *                     CONVOLUTION OPERATIONS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Convolution is fundamental to image processing. It applies a kernel
 * (small matrix) to every pixel in the image.
 *
 * Visual Example (3×3 kernel on image):
 * ════════════════════════════════════════
 *
 *   Image (5×5):              Kernel (3×3):
 *   ┌──┬──┬──┬──┬──┐         ┌──┬──┬──┐
 *   │ 1│ 2│ 3│ 4│ 5│         │k0│k1│k2│
 *   ├──┼──┼──┼──┼──┤         ├──┼──┼──┤
 *   │ 6│ 7│ 8│ 9│10│         │k3│k4│k5│
 *   ├──┼──┼──┼──┼──┤         ├──┼──┼──┤
 *   │11│12│13│14│15│         │k6│k7│k8│
 *   ├──┼──┼──┼──┼──┤         └──┴──┴──┘
 *   │16│17│18│19│20│
 *   ├──┼──┼──┼──┼──┤
 *   │21│22│23│24│25│
 *   └──┴──┴──┴──┴──┘
 *
 *   Output at position (2,2):
 *   result = 1*k0 + 2*k1 + 3*k2 +
 *            6*k3 + 7*k4 + 8*k5 +
 *           11*k6 +12*k7 +13*k8
 *
 * Common Kernels:
 * ──────────────
 *
 * Gaussian Blur (3×3):        Sharpen (3×3):
 * ┌────────────────┐          ┌────────────────┐
 * │ 1/16  2/16  1/16│         │  0  -1   0 │
 * │ 2/16  4/16  2/16│         │ -1   5  -1 │
 * │ 1/16  2/16  1/16│         │  0  -1   0 │
 * └────────────────┘          └────────────────┘
 *
 * Edge Detection (3×3):       Box Filter (3×3):
 * ┌────────────────┐          ┌────────────────┐
 * │ -1  -1  -1 │               │ 1/9  1/9  1/9 │
 * │ -1   8  -1 │               │ 1/9  1/9  1/9 │
 * │ -1  -1  -1 │               │ 1/9  1/9  1/9 │
 * └────────────────┘          └────────────────┘
 */

// Gaussian kernel (5×5) stored in constant memory for fast access
__constant__ float c_gaussianKernel[25];

// Naive 2D convolution (for comparison)
__global__ void convolution2DNaive(unsigned char *input, 
                                   unsigned char *output,
                                   float *kernel, 
                                   int width, int height,
                                   int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int kernelRadius = kernelSize / 2;
        
        // Apply kernel to neighborhood
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int imageRow = row + ky;
                int imageCol = col + kx;
                
                // Handle boundaries with clamping
                imageRow = max(0, min(imageRow, height - 1));
                imageCol = max(0, min(imageCol, width - 1));
                
                int imageIdx = imageRow * width + imageCol;
                int kernelIdx = (ky + kernelRadius) * kernelSize + 
                                (kx + kernelRadius);
                
                sum += input[imageIdx] * kernel[kernelIdx];
            }
        }
        
        // Clamp output to [0, 255]
        output[row * width + col] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

/*
 * Optimized convolution using shared memory
 * 
 * Strategy:
 * ────────
 * 1. Load tile + halo into shared memory
 * 2. Apply kernel using fast shared memory
 * 3. Write result to global memory
 *
 * Memory Access Pattern:
 * ─────────────────────
 *   Global Memory              Shared Memory
 *   ┌────────────────┐         ┌──────────────┐
 *   │                │         │ Halo         │
 *   │  ┌──────────┐  │         │ ┌──────────┐ │
 *   │  │          │  │   →     │ │  Tile    │ │
 *   │  │  Block   │  │         │ │          │ │
 *   │  │          │  │         │ └──────────┘ │
 *   │  └──────────┘  │         │ Halo         │
 *   │                │         └──────────────┘
 *   └────────────────┘         Fast access!
 */

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define SHARED_SIZE (TILE_SIZE + 2 * KERNEL_RADIUS)

__global__ void convolution2DShared(unsigned char *input,
                                    unsigned char *output,
                                    int width, int height) {
    // Shared memory for tile + halo
    __shared__ unsigned char shared[SHARED_SIZE][SHARED_SIZE];
    
    // Global position
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x - KERNEL_RADIUS;
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y - KERNEL_RADIUS;
    
    // Shared memory position
    int sharedCol = threadIdx.x;
    int sharedRow = threadIdx.y;
    
    // Load data into shared memory (including halo)
    if (globalCol >= 0 && globalCol < width && 
        globalRow >= 0 && globalRow < height) {
        shared[sharedRow][sharedCol] = input[globalRow * width + globalCol];
    } else {
        // Pad with zeros for out-of-bounds
        shared[sharedRow][sharedCol] = 0;
    }
    
    __syncthreads();
    
    // Only compute for valid output pixels (not halo)
    if (threadIdx.x >= KERNEL_RADIUS && 
        threadIdx.x < SHARED_SIZE - KERNEL_RADIUS &&
        threadIdx.y >= KERNEL_RADIUS && 
        threadIdx.y < SHARED_SIZE - KERNEL_RADIUS) {
        
        globalCol = blockIdx.x * TILE_SIZE + threadIdx.x - KERNEL_RADIUS;
        globalRow = blockIdx.y * TILE_SIZE + threadIdx.y - KERNEL_RADIUS;
        
        if (globalCol < width && globalRow < height) {
            float sum = 0.0f;
            
            // Apply 5×5 Gaussian kernel from constant memory
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    int sRow = sharedRow + ky;
                    int sCol = sharedCol + kx;
                    int kIdx = (ky + KERNEL_RADIUS) * 5 + (kx + KERNEL_RADIUS);
                    
                    sum += shared[sRow][sCol] * c_gaussianKernel[kIdx];
                }
            }
            
            output[globalRow * width + globalCol] = 
                (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     SOBEL EDGE DETECTION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Sobel operator detects edges by computing image gradients.
 *
 * Sobel X (horizontal edges):    Sobel Y (vertical edges):
 * ┌──────────────┐               ┌──────────────┐
 * │ -1   0   1 │                 │ -1  -2  -1 │
 * │ -2   0   2 │                 │  0   0   0 │
 * │ -1   0   1 │                 │  1   2   1 │
 * └──────────────┘               └──────────────┘
 *
 * Edge Magnitude:
 * magnitude = sqrt(Gx² + Gy²)
 *
 * Visual Example:
 * ──────────────
 * Input:        Gx (horizontal):  Gy (vertical):   Magnitude:
 * ┌───────┐     ┌───────┐        ┌───────┐        ┌───────┐
 * │       │     │   0   │        │   0   │        │   0   │
 * │   █   │  →  │ ███   │        │ ███   │   →    │ ███   │
 * │       │     │   0   │        │   0   │        │   0   │
 * └───────┘     └───────┘        └───────┘        └───────┘
 */

__global__ void sobelEdgeDetection(unsigned char *input,
                                   unsigned char *output,
                                   int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        // Sobel X kernel
        float Gx = 0.0f;
        Gx += -1 * input[(row - 1) * width + (col - 1)];
        Gx +=  0 * input[(row - 1) * width + (col    )];
        Gx +=  1 * input[(row - 1) * width + (col + 1)];
        Gx += -2 * input[(row    ) * width + (col - 1)];
        Gx +=  0 * input[(row    ) * width + (col    )];
        Gx +=  2 * input[(row    ) * width + (col + 1)];
        Gx += -1 * input[(row + 1) * width + (col - 1)];
        Gx +=  0 * input[(row + 1) * width + (col    )];
        Gx +=  1 * input[(row + 1) * width + (col + 1)];
        
        // Sobel Y kernel
        float Gy = 0.0f;
        Gy += -1 * input[(row - 1) * width + (col - 1)];
        Gy += -2 * input[(row - 1) * width + (col    )];
        Gy += -1 * input[(row - 1) * width + (col + 1)];
        Gy +=  0 * input[(row    ) * width + (col - 1)];
        Gy +=  0 * input[(row    ) * width + (col    )];
        Gy +=  0 * input[(row    ) * width + (col + 1)];
        Gy +=  1 * input[(row + 1) * width + (col - 1)];
        Gy +=  2 * input[(row + 1) * width + (col    )];
        Gy +=  1 * input[(row + 1) * width + (col + 1)];
        
        // Compute magnitude
        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        
        // Clamp to [0, 255]
        output[row * width + col] = 
            (unsigned char)fminf(magnitude, 255.0f);
    }
}

/*
 * Optimized Sobel using shared memory
 */
__global__ void sobelEdgeDetectionShared(unsigned char *input,
                                         unsigned char *output,
                                         int width, int height) {
    __shared__ unsigned char shared[18][18];  // 16×16 + 2-pixel border
    
    int globalCol = blockIdx.x * 16 + threadIdx.x - 1;
    int globalRow = blockIdx.y * 16 + threadIdx.y - 1;
    int sharedCol = threadIdx.x;
    int sharedRow = threadIdx.y;
    
    // Load into shared memory
    if (globalCol >= 0 && globalCol < width && 
        globalRow >= 0 && globalRow < height) {
        shared[sharedRow][sharedCol] = input[globalRow * width + globalCol];
    } else {
        shared[sharedRow][sharedCol] = 0;
    }
    
    __syncthreads();
    
    // Compute Sobel for interior pixels only
    if (threadIdx.x >= 1 && threadIdx.x < 17 &&
        threadIdx.y >= 1 && threadIdx.y < 17) {
        
        globalCol = blockIdx.x * 16 + threadIdx.x - 1;
        globalRow = blockIdx.y * 16 + threadIdx.y - 1;
        
        if (globalCol < width && globalRow < height) {
            // Apply Sobel using shared memory (much faster!)
            float Gx = -shared[sharedRow-1][sharedCol-1] + 
                       shared[sharedRow-1][sharedCol+1] -
                    2*shared[sharedRow  ][sharedCol-1] + 
                    2*shared[sharedRow  ][sharedCol+1] -
                       shared[sharedRow+1][sharedCol-1] + 
                       shared[sharedRow+1][sharedCol+1];
            
            float Gy = -shared[sharedRow-1][sharedCol-1] - 
                    2*shared[sharedRow-1][sharedCol  ] - 
                       shared[sharedRow-1][sharedCol+1] +
                       shared[sharedRow+1][sharedCol-1] + 
                    2*shared[sharedRow+1][sharedCol  ] + 
                       shared[sharedRow+1][sharedCol+1];
            
            float magnitude = sqrtf(Gx * Gx + Gy * Gy);
            output[globalRow * width + globalCol] = 
                (unsigned char)fminf(magnitude, 255.0f);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     HISTOGRAM COMPUTATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Histogram counts the frequency of each intensity value (0-255).
 *
 * Example:
 * ───────
 * Image:        Histogram:
 * ┌───────┐     ┌─────────────────┐
 * │2 2 5 8│     │ 0: ░░░░░░      │
 * │2 5 8 8│  →  │ 1: ░           │
 * │5 5 8 9│     │ 2: ░░░░        │
 * │8 9 9 9│     │ 3: ░           │
 * └───────┘     │ 4:             │
 *               │ 5: ░░░░        │
 *               │ 6:             │
 *               │ 7:             │
 *               │ 8: ░░░░░       │
 *               │ 9: ░░░░        │
 *               └─────────────────┘
 *
 * Uses atomic operations to avoid race conditions when multiple threads
 * update the same histogram bin.
 */

#define NUM_BINS 256

__global__ void computeHistogram(unsigned char *image,
                                 unsigned int *histogram,
                                 int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        unsigned char pixel = image[row * width + col];
        atomicAdd(&histogram[pixel], 1);
    }
}

/*
 * Optimized histogram using shared memory to reduce atomic contention
 * on global memory.
 *
 * Strategy:
 * ────────
 * 1. Each block has a local histogram in shared memory
 * 2. Threads update local histogram (faster atomics)
 * 3. Merge local histograms into global histogram
 */

__global__ void computeHistogramShared(unsigned char *image,
                                       unsigned int *histogram,
                                       int width, int height) {
    // Shared histogram for this block
    __shared__ unsigned int sharedHist[NUM_BINS];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    
    // Initialize shared histogram
    for (int i = tid; i < NUM_BINS; i += blockSize) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    // Compute histogram in shared memory
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        unsigned char pixel = image[row * width + col];
        atomicAdd(&sharedHist[pixel], 1);
    }
    __syncthreads();
    
    // Merge into global histogram
    for (int i = tid; i < NUM_BINS; i += blockSize) {
        if (sharedHist[i] > 0) {
            atomicAdd(&histogram[i], sharedHist[i]);
        }
    }
}

/*
 * Histogram Equalization
 * ─────────────────────
 * Improves image contrast by redistributing intensity values.
 *
 * Steps:
 * 1. Compute histogram
 * 2. Compute cumulative distribution function (CDF)
 * 3. Normalize CDF to [0, 255]
 * 4. Map each pixel through the normalized CDF
 *
 * Visual Effect:
 * ─────────────
 * Before:           After:
 * ┌─────────┐       ┌─────────┐
 * │░░░░░░░░░│       │░░▒▒▓▓███│  Better contrast
 * │░░░░░░░░░│  →    │░░▒▒▓▓███│  More detail visible
 * │▒▒▒▒▒▒▒▒▒│       │░░▒▒▓▓███│
 * └─────────┘       └─────────┘
 */

__global__ void histogramEqualization(unsigned char *input,
                                      unsigned char *output,
                                      unsigned int *cdf,
                                      int width, int height,
                                      int totalPixels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        unsigned char pixel = input[idx];
        
        // Map through CDF
        // Formula: output = (cdf[pixel] - cdf_min) * 255 / (total - cdf_min)
        int cdfMin = cdf[0];
        int newValue = ((cdf[pixel] - cdfMin) * 255) / (totalPixels - cdfMin);
        
        output[idx] = (unsigned char)max(0, min(newValue, 255));
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     BILATERAL FILTER
 * ═══════════════════════════════════════════════════════════════════
 *
 * Bilateral filter smooths images while preserving edges.
 * It uses both spatial and intensity differences.
 *
 * Unlike Gaussian blur which blurs everything:
 * ───────────────────────────────────────────
 *
 * Gaussian Blur:        Bilateral Filter:
 * ┌─────────┐           ┌─────────┐
 * │░░░░░░░░░│           │░░░░░███░│  Edges preserved!
 * │░░░░░░░░░│           │░░░░░███░│
 * │▓▓▓▓▓▓▓▓▓│  →        │▓▓▓▓▓███▓│
 * │▓▓▓▓▓▓▓▓▓│           │▓▓▓▓▓███▓│
 * └─────────┘           └─────────┘
 * Everything blurred    Smooth + sharp edges
 *
 * Weight Formula:
 * w(i,j,k,l) = exp(-(distance² / (2*σ_spatial²))) *
 *              exp(-(intensity_diff² / (2*σ_range²)))
 *
 * Distance weight: closer pixels have higher weight
 * Intensity weight: similar colors have higher weight
 */

__global__ void bilateralFilter(unsigned char *input,
                                unsigned char *output,
                                int width, int height,
                                int radius,
                                float sigmaSpace,
                                float sigmaColor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int centerIdx = row * width + col;
        float centerPixel = input[centerIdx];
        
        float sum = 0.0f;
        float weightSum = 0.0f;
        
        // Apply bilateral filter over neighborhood
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int neighborRow = row + dy;
                int neighborCol = col + dx;
                
                // Boundary check
                if (neighborRow >= 0 && neighborRow < height &&
                    neighborCol >= 0 && neighborCol < width) {
                    
                    int neighborIdx = neighborRow * width + neighborCol;
                    float neighborPixel = input[neighborIdx];
                    
                    // Spatial distance weight
                    float spatialDist = sqrtf(dx * dx + dy * dy);
                    float spatialWeight = expf(-(spatialDist * spatialDist) / 
                                              (2 * sigmaSpace * sigmaSpace));
                    
                    // Intensity difference weight
                    float colorDist = fabsf(neighborPixel - centerPixel);
                    float colorWeight = expf(-(colorDist * colorDist) / 
                                            (2 * sigmaColor * sigmaColor));
                    
                    // Combined weight
                    float weight = spatialWeight * colorWeight;
                    
                    sum += neighborPixel * weight;
                    weightSum += weight;
                }
            }
        }
        
        // Normalize and write output
        output[centerIdx] = (unsigned char)(sum / weightSum);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     MEDIAN FILTER
 * ═══════════════════════════════════════════════════════════════════
 *
 * Median filter is excellent for removing salt-and-pepper noise.
 * It replaces each pixel with the median of its neighborhood.
 *
 * Example (3×3 window):
 * ────────────────────
 * Input:              Sort values:      Output:
 * ┌───────┐          [5,10,12,15,      ┌───┐
 * │ 5│10│255│         20,22,30,42,255] │   │
 * ├──┼──┼───┤                           │ 20│  Median value
 * │15│20│ 22│    →   Median = 20   →   │   │
 * ├──┼──┼───┤                           └───┘
 * │30│42│255│
 * └───────────┘
 *
 * Effect on noise:
 * ───────────────
 * Noisy:            Denoised:
 * ┌──────────┐      ┌──────────┐
 * │░░█░░░█░░░│      │░░░░░░░░░░│  Noise removed!
 * │░░░█░░░░█░│  →   │░░░░░░░░░░│  Edges preserved!
 * │█░░░░█░░░░│      │░░░░░░░░░░│
 * └──────────┘      └──────────┘
 */

// Bubble sort for median finding (simple for small windows)
__device__ void bubbleSort(unsigned char *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                unsigned char temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__global__ void medianFilter(unsigned char *input,
                             unsigned char *output,
                             int width, int height,
                             int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Collect neighborhood pixels
        int windowSize = (2 * radius + 1) * (2 * radius + 1);
        unsigned char window[25];  // Max 5×5 window
        int count = 0;
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int neighborRow = row + dy;
                int neighborCol = col + dx;
                
                if (neighborRow >= 0 && neighborRow < height &&
                    neighborCol >= 0 && neighborCol < width) {
                    window[count++] = input[neighborRow * width + neighborCol];
                }
            }
        }
        
        // Sort and find median
        bubbleSort(window, count);
        output[row * width + col] = window[count / 2];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     BOX FILTER (SIMPLE BLUR)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Box filter is the simplest blur - averages all pixels in a window.
 *
 * Separable Property:
 * ──────────────────
 * 2D box filter can be decomposed into two 1D filters:
 * 
 * One 2D pass (N² operations):
 * ┌───────────┐
 * │ 1  1  1 │
 * │ 1  1  1 │ / 9
 * │ 1  1  1 │
 * └───────────┘
 *
 * Two 1D passes (2N operations - much faster!):
 * Horizontal:      Vertical:
 * ┌───────────┐    ┌───┐
 * │ 1  1  1 │    │ 1 │
 * └───────────┘    │ 1 │ / 3
 *                  │ 1 │
 *                  └───┘
 */

// Horizontal pass of separable box filter
__global__ void boxFilterHorizontal(unsigned char *input,
                                    float *output,
                                    int width, int height,
                                    int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int count = 0;
        
        for (int dx = -radius; dx <= radius; dx++) {
            int neighborCol = col + dx;
            if (neighborCol >= 0 && neighborCol < width) {
                sum += input[row * width + neighborCol];
                count++;
            }
        }
        
        output[row * width + col] = sum / count;
    }
}

// Vertical pass of separable box filter
__global__ void boxFilterVertical(float *input,
                                  unsigned char *output,
                                  int width, int height,
                                  int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int count = 0;
        
        for (int dy = -radius; dy <= radius; dy++) {
            int neighborRow = row + dy;
            if (neighborRow >= 0 && neighborRow < height) {
                sum += input[neighborRow * width + col];
                count++;
            }
        }
        
        output[row * width + col] = (unsigned char)(sum / count);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 */

// Generate synthetic test image (checkerboard pattern)
void generateTestImage(unsigned char *image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blockSize = 32;
            int checkX = (x / blockSize) % 2;
            int checkY = (y / blockSize) % 2;
            image[y * width + x] = (checkX ^ checkY) ? 255 : 64;
        }
    }
    
    // Add some noise for testing filters
    for (int i = 0; i < width * height / 100; i++) {
        int idx = rand() % (width * height);
        image[idx] = (rand() % 2) * 255;  // Salt and pepper noise
    }
}

// Initialize Gaussian kernel in constant memory
void initGaussianKernel(float sigma) {
    float kernel[25];
    float sum = 0.0f;
    int radius = 2;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + radius) * 5 + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < 25; i++) {
        kernel[i] /= sum;
    }
    
    // Copy to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussianKernel, kernel, 
                                  25 * sizeof(float)));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║         CUDA Tutorial: Image Processing               ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // Image dimensions
    const int width = 1920;
    const int height = 1080;
    const int imageSize = width * height;
    const size_t bytes = imageSize * sizeof(unsigned char);
    
    printf("Image size: %dx%d (%zu MB)\n\n", width, height, 
           bytes / (1024 * 1024));
    
    // Allocate host memory
    unsigned char *h_input = (unsigned char*)malloc(bytes);
    unsigned char *h_output = (unsigned char*)malloc(bytes);
    
    // Generate test image
    generateTestImage(h_input, width, height);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Initialize Gaussian kernel
    initGaussianKernel(1.5f);
    
    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Gaussian Blur (Naive vs Shared Memory)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Gaussian Blur\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Shared memory version
    dim3 sharedBlockSize(SHARED_SIZE, SHARED_SIZE);
    dim3 sharedGridSize((width + TILE_SIZE - 1) / TILE_SIZE,
                        (height + TILE_SIZE - 1) / TILE_SIZE);
    
    CUDA_CHECK(cudaEventRecord(start));
    convolution2DShared<<<sharedGridSize, sharedBlockSize>>>(d_input, d_output, 
                                                              width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float blurTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&blurTime, start, stop));
    
    printf("Gaussian Blur (5×5 kernel):\n");
    printf("  Time: %.3f ms\n", blurTime);
    printf("  Throughput: %.2f Mpixels/s\n\n", 
           imageSize / (blurTime * 1000.0f));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Sobel Edge Detection
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Sobel Edge Detection\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    sobelEdgeDetectionShared<<<gridSize, blockSize>>>(d_input, d_output, 
                                                       width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float sobelTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&sobelTime, start, stop));
    
    printf("Sobel Edge Detection:\n");
    printf("  Time: %.3f ms\n", sobelTime);
    printf("  Throughput: %.2f Mpixels/s\n\n", 
           imageSize / (sobelTime * 1000.0f));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Histogram Computation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Histogram Computation\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    unsigned int *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
    
    CUDA_CHECK(cudaEventRecord(start));
    computeHistogramShared<<<gridSize, blockSize>>>(d_input, d_histogram, 
                                                     width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float histTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&histTime, start, stop));
    
    printf("Histogram Computation:\n");
    printf("  Time: %.3f ms\n", histTime);
    printf("  Throughput: %.2f Mpixels/s\n\n", 
           imageSize / (histTime * 1000.0f));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 4: Bilateral Filter
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 4: Bilateral Filter\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    bilateralFilter<<<gridSize, blockSize>>>(d_input, d_output, 
                                              width, height, 5, 3.0f, 50.0f);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float bilateralTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&bilateralTime, start, stop));
    
    printf("Bilateral Filter (radius=5):\n");
    printf("  Time: %.3f ms\n", bilateralTime);
    printf("  Throughput: %.2f Mpixels/s\n\n", 
           imageSize / (bilateralTime * 1000.0f));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 5: Median Filter
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 5: Median Filter\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    medianFilter<<<gridSize, blockSize>>>(d_input, d_output, 
                                          width, height, 1);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float medianTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&medianTime, start, stop));
    
    printf("Median Filter (3×3 window):\n");
    printf("  Time: %.3f ms\n", medianTime);
    printf("  Throughput: %.2f Mpixels/s\n\n", 
           imageSize / (medianTime * 1000.0f));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Performance Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Performance Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("┌─────────────────────┬────────────┬──────────────┐\n");
    printf("│ Operation           │ Time (ms)  │ Throughput   │\n");
    printf("├─────────────────────┼────────────┼──────────────┤\n");
    printf("│ Gaussian Blur       │ %9.3f  │ %7.2f MP/s │\n", 
           blurTime, imageSize / (blurTime * 1000.0f));
    printf("│ Sobel Edge Detect   │ %9.3f  │ %7.2f MP/s │\n", 
           sobelTime, imageSize / (sobelTime * 1000.0f));
    printf("│ Histogram           │ %9.3f  │ %7.2f MP/s │\n", 
           histTime, imageSize / (histTime * 1000.0f));
    printf("│ Bilateral Filter    │ %9.3f  │ %7.2f MP/s │\n", 
           bilateralTime, imageSize / (bilateralTime * 1000.0f));
    printf("│ Median Filter       │ %9.3f  │ %7.2f MP/s │\n", 
           medianTime, imageSize / (medianTime * 1000.0f));
    printf("└─────────────────────┴────────────┴──────────────┘\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_histogram));
    
    free(h_input);
    free(h_output);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Shared memory provides 2-5x speedup               ║\n");
    printf("║ 2. Separable filters reduce complexity (N² → 2N)     ║\n");
    printf("║ 3. Bilateral filter preserves edges while smoothing  ║\n");
    printf("║ 4. Median filter excels at salt-pepper noise         ║\n");
    printf("║ 5. Histogram benefits from shared memory reduction   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement Canny edge detection (multi-stage)
 * 2. Add morphological operations (erosion, dilation)
 * 3. Implement anisotropic diffusion filter
 * 4. Add Harris corner detection
 * 5. Implement separable Gaussian blur
 * 6. Add adaptive histogram equalization (CLAHE)
 * 7. Implement non-local means denoising
 * 8. Add image pyramid generation
 *
 * ═══════════════════════════════════════════════════════════════════
 */

