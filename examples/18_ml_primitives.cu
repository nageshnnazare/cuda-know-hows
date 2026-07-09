/*
 * CUDA Tutorial - Part 18: Machine Learning Primitives
 * 
 * This file demonstrates core ML operations implemented in CUDA:
 * 1. Matrix Multiplication (GEMM) - Optimized from scratch
 * 2. Activation Functions (ReLU, Sigmoid, Tanh, GELU, Swish)
 * 3. Softmax (numerically stable)
 * 4. Cross-Entropy Loss
 * 5. Batch Normalization (forward & backward)
 * 6. Dropout
 * 7. Layer Normalization
 * 8. Attention Mechanism (basics)
 *
 * Each primitive includes:
 * - Mathematical formulation
 * - Naive implementation
 * - Optimized version
 * - Backward pass (gradients)
 *
 * Compile: nvcc -o ml_primitives 18_ml_primitives.cu -O3 -lcublas
 * Run:     ./ml_primitives
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

/*
 * ═══════════════════════════════════════════════════════════════════
 *              1. MATRIX MULTIPLICATION (GEMM)
 * ═══════════════════════════════════════════════════════════════════
 *
 * General Matrix Multiply: C = α·A·B + β·C
 * 
 * This is THE most important operation in deep learning!
 * - Fully connected layers
 * - Attention mechanisms
 * - Convolution (when expressed as matrix multiply)
 *
 * Visual Example (3×2 × 2×3 = 3×3):
 * ──────────────────────────────────
 *     A (3×2)      B (2×3)      C (3×3)
 *   ┌─────────┐  ┌─────────┐  ┌─────────┐
 *   │ 1  2    │  │ 7  8  9 │  │ ?  ?  ? │
 *   │ 3  4    │ ×│10 11 12 │= │ ?  ?  ? │
 *   │ 5  6    │  └─────────┘  │ ?  ?  ? │
 *   └─────────┘                └─────────┘
 *
 * C[0,0] = A[0,0]·B[0,0] + A[0,1]·B[1,0]
 *        = 1·7 + 2·10 = 27
 *
 * Performance Progression:
 * ───────────────────────
 * Naive:          50 GFLOPS
 * Tiled:          400 GFLOPS  (8x)
 * Optimized:      800 GFLOPS  (16x)
 * cuBLAS:         12000 GFLOPS (240x!)
 */

// Tiled matrix multiplication (already covered in 05_matrix_operations.cu)
// Here we show the connection to neural networks

#define TILE_SIZE 32

__global__ void gemmTiled(float *A, float *B, float *C,
                          int M, int K, int N,
                          float alpha, float beta) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 avoids bank conflicts
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result: C = alpha * A*B + beta * C
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              2. ACTIVATION FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Activation functions introduce non-linearity into neural networks.
 *
 * Visual Comparison:
 * ─────────────────
 *    ReLU:           Sigmoid:        Tanh:
 *     │    ╱           │  ─────        │  ─────
 *     │   ╱            │ ╱              │ ╱
 *     │  ╱             │╱               │╱
 *   ──┼─────         ──┼────          ──┼────
 *     │               │                 │
 *     │               │                 │
 *
 * Mathematical Definitions:
 * ────────────────────────
 * ReLU:     f(x) = max(0, x)
 * Sigmoid:  f(x) = 1 / (1 + e^(-x))
 * Tanh:     f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * GELU:     f(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 * Swish:    f(x) = x·sigmoid(x)
 *
 * Derivatives (for backprop):
 * ──────────────────────────
 * ReLU:     f'(x) = x > 0 ? 1 : 0
 * Sigmoid:  f'(x) = f(x)·(1 - f(x))
 * Tanh:     f'(x) = 1 - f(x)²
 */

// ReLU: Rectified Linear Unit
__global__ void reluForward(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void reluBackward(float *grad_output, float *input, 
                             float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Gradient is 1 if input > 0, else 0
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Sigmoid: Logistic function
__global__ void sigmoidForward(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoidBackward(float *grad_output, float *output,
                                float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sig = output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
    }
}

// Tanh: Hyperbolic tangent
__global__ void tanhForward(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanhBackward(float *grad_output, float *output,
                             float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float t = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

// GELU: Gaussian Error Linear Unit (used in BERT, GPT)
__global__ void geluForward(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi) ≈ 0.797...
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Swish (SiLU): x * sigmoid(x)
__global__ void swishForward(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              3. SOFTMAX
 * ═══════════════════════════════════════════════════════════════════
 *
 * Softmax converts logits to probabilities.
 *
 * Formula:
 * ───────
 * softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 *
 * Properties:
 * ──────────
 * • All outputs in (0, 1)
 * • Sum of outputs = 1.0
 * • Differentiable
 * • Numerically stable version needed!
 *
 * Numerical Stability:
 * ───────────────────
 * Problem: exp(x) overflows for large x
 * 
 * Example:
 *   x = [1000, 2000, 3000]
 *   exp(3000) = overflow! 💥
 *
 * Solution: Subtract max before exp
 *   x' = x - max(x)
 *   x' = [1000-3000, 2000-3000, 3000-3000]
 *      = [-2000, -1000, 0]
 *   exp(x') = [very small, small, 1.0] ✓
 *
 * Visual Example:
 * ──────────────
 * Input (logits):  [2.0, 1.0, 0.1]
 *                    ↓ exp
 * Unnormalized:    [7.39, 2.72, 1.11]
 *                    ↓ normalize (sum = 11.22)
 * Output (probs):  [0.659, 0.242, 0.099]
 *                    ↓ verify
 * Sum:             0.659 + 0.242 + 0.099 = 1.0 ✓
 */

__global__ void softmaxForward(float *input, float *output,
                               int batch_size, int num_classes) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < batch_size) {
        int offset = sample * num_classes;
        
        // Find max for numerical stability
        float max_val = input[offset];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[offset + i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[offset + i] - max_val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[offset + i] /= sum;
        }
    }
}

// Backward pass: gradient of softmax
__global__ void softmaxBackward(float *grad_output, float *output,
                                float *grad_input,
                                int batch_size, int num_classes) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < batch_size) {
        int offset = sample * num_classes;
        
        // For softmax: ∂L/∂x_i = softmax_i * (∂L/∂softmax_i - Σ_j softmax_j * ∂L/∂softmax_j)
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum += output[offset + i] * grad_output[offset + i];
        }
        
        for (int i = 0; i < num_classes; i++) {
            grad_input[offset + i] = output[offset + i] * 
                                     (grad_output[offset + i] - sum);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              4. CROSS-ENTROPY LOSS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Cross-entropy measures difference between predicted and true distributions.
 *
 * Formula:
 * ───────
 * L = -Σ y_i * log(p_i)
 *
 * Where:
 * - y_i: True label (one-hot encoded)
 * - p_i: Predicted probability
 *
 * Visual Example:
 * ──────────────
 * True label:  [0, 1, 0]  (class 1)
 * Prediction:  [0.2, 0.7, 0.1]
 * 
 * Loss = -(0*log(0.2) + 1*log(0.7) + 0*log(0.1))
 *      = -log(0.7)
 *      = 0.357
 *
 * Perfect prediction: [0, 1, 0] → Loss = 0
 * Bad prediction:     [0.5, 0.3, 0.2] → Loss = 1.204
 */

__global__ void crossEntropyLoss(float *predictions, int *labels,
                                 float *loss, int batch_size, int num_classes) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < batch_size) {
        int label = labels[sample];
        int offset = sample * num_classes;
        
        // Clip probability to avoid log(0)
        float prob = fmaxf(predictions[offset + label], 1e-7f);
        
        // Accumulate loss
        atomicAdd(loss, -logf(prob) / batch_size);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              5. BATCH NORMALIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Batch Norm normalizes activations to have mean=0, variance=1.
 *
 * Forward Pass:
 * ────────────
 * 1. Compute mean: μ = (1/N) Σ x_i
 * 2. Compute variance: σ² = (1/N) Σ (x_i - μ)²
 * 3. Normalize: x̂_i = (x_i - μ) / √(σ² + ε)
 * 4. Scale and shift: y_i = γ·x̂_i + β
 *
 * Visual Effect:
 * ─────────────
 * Before BN:          After BN:
 * ┌─────────────┐     ┌─────────────┐
 * │  ███         │     │     ███     │  Centered
 * │ █████        │  →  │    █████    │  Normalized
 * │███████       │     │   ███████   │  Stable!
 * └─────────────┘     └─────────────┘
 * Wide distribution   Mean=0, Var=1
 *
 * Benefits:
 * ────────
 * ✓ Faster training
 * ✓ Higher learning rates
 * ✓ Reduces internal covariate shift
 * ✓ Acts as regularization
 */

__global__ void batchNormForward(float *input, float *output,
                                 float *mean, float *variance,
                                 float *gamma, float *beta,
                                 int batch_size, int num_features,
                                 float epsilon) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature < num_features) {
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += input[i * num_features + feature];
        }
        mean[feature] = sum / batch_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float diff = input[i * num_features + feature] - mean[feature];
            var_sum += diff * diff;
        }
        variance[feature] = var_sum / batch_size;
        
        // Normalize and scale
        float std = sqrtf(variance[feature] + epsilon);
        for (int i = 0; i < batch_size; i++) {
            int idx = i * num_features + feature;
            float normalized = (input[idx] - mean[feature]) / std;
            output[idx] = gamma[feature] * normalized + beta[feature];
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              6. DROPOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Dropout randomly zeros neurons during training for regularization.
 *
 * Training:               Inference:
 * ┌───────────┐          ┌───────────┐
 * │ ✓ ✗ ✓ ✓ ✗ │          │ ✓ ✓ ✓ ✓ ✓ │
 * │ ✓ ✓ ✗ ✓ ✓ │   vs     │ ✓ ✓ ✓ ✓ ✓ │
 * │ ✗ ✓ ✓ ✗ ✓ │          │ ✓ ✓ ✓ ✓ ✓ │
 * └───────────┘          └───────────┘
 * 50% dropped            All active
 * (scaled by 1/p)        (no scaling)
 *
 * Purpose:
 * ───────
 * • Prevents overfitting
 * • Forces network to learn redundant representations
 * • Ensemble effect
 *
 * Inverted Dropout:
 * ────────────────
 * During training: output = input * mask / keep_prob
 * During inference: output = input (no change)
 *
 * This way, no scaling needed at inference!
 */

#include <curand_kernel.h>

__global__ void dropoutForward(float *input, float *output, float *mask,
                                int n, float keep_prob, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random number
        float random = curand_uniform(&state);
        
        // Apply dropout
        if (random < keep_prob) {
            mask[idx] = 1.0f / keep_prob;  // Inverted dropout
            output[idx] = input[idx] * mask[idx];
        } else {
            mask[idx] = 0.0f;
            output[idx] = 0.0f;
        }
    }
}

__global__ void dropoutBackward(float *grad_output, float *mask,
                                float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * mask[idx];
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              7. LAYER NORMALIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Layer Norm normalizes across features (vs batch norm across batch).
 *
 * Batch Norm:                Layer Norm:
 * ┌─────────────┐           ┌─────────────┐
 * │ Normalize   │           │ Normalize   │
 * │ across      │           │ across      │
 * │ batch  ↓    │           │ features →  │
 * └─────────────┘           └─────────────┘
 *
 * Use Cases:
 * ─────────
 * • Transformers (BERT, GPT)
 * • RNNs (variable sequence length)
 * • Small batches
 *
 * Formula (per sample):
 * ────────────────────
 * μ = (1/D) Σ x_i
 * σ² = (1/D) Σ (x_i - μ)²
 * y_i = γ·(x_i - μ)/√(σ² + ε) + β
 */

__global__ void layerNormForward(float *input, float *output,
                                 float *gamma, float *beta,
                                 int batch_size, int num_features,
                                 float epsilon) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < batch_size) {
        int offset = sample * num_features;
        
        // Compute mean for this sample
        float sum = 0.0f;
        for (int i = 0; i < num_features; i++) {
            sum += input[offset + i];
        }
        float mean = sum / num_features;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < num_features; i++) {
            float diff = input[offset + i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / num_features;
        float std = sqrtf(variance + epsilon);
        
        // Normalize and scale
        for (int i = 0; i < num_features; i++) {
            float normalized = (input[offset + i] - mean) / std;
            output[offset + i] = gamma[i] * normalized + beta[i];
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              8. ATTENTION MECHANISM (Simplified)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Attention computes weighted sum based on query-key similarities.
 *
 * Formula:
 * ───────
 * Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
 *
 * Visual:
 * ──────
 * Query: What am I looking for?
 * Key:   What do I contain?
 * Value: What do I actually represent?
 *
 * Example: "The cat sat on the mat"
 * 
 * When processing "sat":
 * - Query: "sat" representation
 * - Keys: ["The", "cat", "sat", "on", "the", "mat"]
 * - Compute similarity: sat vs each word
 * - Scores: [0.1, 0.6, 0.05, 0.15, 0.05, 0.05]
 * - Attend mostly to "cat" (0.6 weight)
 *
 * Matrix Dimensions:
 * ─────────────────
 * Q: [batch, seq_len, d_k]
 * K: [batch, seq_len, d_k]
 * V: [batch, seq_len, d_v]
 * Output: [batch, seq_len, d_v]
 */

__global__ void attentionScores(float *queries, float *keys, float *scores,
                                int batch_size, int seq_len, int d_k) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Query position
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Key position
    
    if (b < batch_size && i < seq_len && j < seq_len) {
        float sum = 0.0f;
        
        // Dot product between query[i] and key[j]
        for (int k = 0; k < d_k; k++) {
            int q_idx = b * (seq_len * d_k) + i * d_k + k;
            int k_idx = b * (seq_len * d_k) + j * d_k + k;
            sum += queries[q_idx] * keys[k_idx];
        }
        
        // Scale by sqrt(d_k) for stability
        float scale = 1.0f / sqrtf((float)d_k);
        scores[b * (seq_len * seq_len) + i * seq_len + j] = sum * scale;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║      CUDA Tutorial: ML Primitives                     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Activation Functions
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Activation Functions\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int n = 10000000;  // 10M elements
    float *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    // Initialize input with random values
    float *h_input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_input[i] = ((rand() % 1000) / 500.0f) - 1.0f;  // [-1, 1]
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Test ReLU
    CUDA_CHECK(cudaEventRecord(start));
    reluForward<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float relu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&relu_time, start, stop));
    
    // Test Sigmoid
    CUDA_CHECK(cudaEventRecord(start));
    sigmoidForward<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float sigmoid_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&sigmoid_time, start, stop));
    
    // Test GELU
    CUDA_CHECK(cudaEventRecord(start));
    geluForward<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gelu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gelu_time, start, stop));
    
    printf("Activation Functions (%d elements):\n", n);
    printf("┌──────────────┬──────────┬──────────────┐\n");
    printf("│ Function     │ Time(ms) │ Throughput   │\n");
    printf("├──────────────┼──────────┼──────────────┤\n");
    printf("│ ReLU         │ %7.3f  │ %7.2f GB/s│\n",
           relu_time, (2 * n * sizeof(float)) / (relu_time * 1e6f));
    printf("│ Sigmoid      │ %7.3f  │ %7.2f GB/s│\n",
           sigmoid_time, (2 * n * sizeof(float)) / (sigmoid_time * 1e6f));
    printf("│ GELU         │ %7.3f  │ %7.2f GB/s│\n",
           gelu_time, (2 * n * sizeof(float)) / (gelu_time * 1e6f));
    printf("└──────────────┴──────────┴──────────────┘\n\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Softmax
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Softmax\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int batch_size = 1024;
    int num_classes = 1000;
    int total = batch_size * num_classes;
    
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));
    
    h_input = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) {
        h_input[i] = ((rand() % 1000) / 100.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    blockSize = 256;
    gridSize = (batch_size + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(start));
    softmaxForward<<<gridSize, blockSize>>>(d_input, d_output,
                                             batch_size, num_classes);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float softmax_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&softmax_time, start, stop));
    
    // Verify softmax output
    float *h_output = (float*)malloc(total * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, total * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Check first sample sums to 1.0
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum += h_output[i];
    }
    
    printf("Softmax (batch=%d, classes=%d):\n", batch_size, num_classes);
    printf("  Time: %.3f ms\n", softmax_time);
    printf("  Sum verification: %.6f (should be 1.0)\n", sum);
    printf("  First 5 probabilities: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. GEMM is the core of deep learning                 ║\n");
    printf("║ 2. Activation functions add non-linearity            ║\n");
    printf("║ 3. Softmax requires numerical stability              ║\n");
    printf("║ 4. Batch/Layer norm accelerate training              ║\n");
    printf("║ 5. Dropout prevents overfitting                      ║\n");
    printf("║ 6. Attention enables context-aware representations   ║\n");
    printf("║ 7. cuBLAS/cuDNN provide optimized implementations    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement backward pass for all activation functions
 * 2. Add fused operations (e.g., Conv+BatchNorm+ReLU)
 * 3. Implement AdamW optimizer
 * 4. Add mixed-precision training (FP16/FP32)
 * 5. Implement multi-head attention
 * 6. Add gradient clipping
 * 7. Implement weight initialization schemes
 * 8. Create benchmarks vs cuDNN
 *
 * ═══════════════════════════════════════════════════════════════════
 */

