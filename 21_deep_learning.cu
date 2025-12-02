/*
 * CUDA Tutorial - Part 21: Deep Learning from Scratch
 * 
 * This file demonstrates deep learning implementations in CUDA:
 * 1. Linear Regression (single neuron, MSE loss)
 * 2. Logistic Regression (binary classification, sigmoid activation)
 * 3. Multi-class Classification (softmax, cross-entropy loss)
 * 4. Feedforward Neural Network (hidden layers, backpropagation)
 * 5. Convolutional Neural Network (CNN for image classification)
 * 6. Simple Object Detection (bounding box prediction)
 *
 * Each section builds on previous concepts with:
 * - Clear mathematical explanations
 * - Network architecture diagrams
 * - Forward and backward pass implementations
 * - Training loop with GPU acceleration
 *
 * Compile: nvcc -o deep_learning 21_deep_learning.cu -O3 -arch=sm_70
 * Run:     ./deep_learning
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
 *                    PART 1: LINEAR REGRESSION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Linear Regression finds a line that best fits data points.
 *
 * Mathematical Model:
 * ──────────────────
 * ŷ = w·x + b
 *
 * Where:
 * - x: input feature
 * - w: weight (slope)
 * - b: bias (intercept)
 * - ŷ: predicted output
 *
 * Visual Example:
 * ──────────────
 *   y
 *   │     •
 *   │        •
 *   │   •      ŷ = w·x + b
 *   │      •    /
 *   │   •     /
 *   │      • /
 *   │     •/
 *   │    /•
 *   │  /
 *   │/________________ x
 *
 * Loss Function (Mean Squared Error):
 * ───────────────────────────────────
 * L = (1/n) Σ(ŷᵢ - yᵢ)²
 *
 * Gradient Descent Update:
 * ───────────────────────
 * w = w - α·∂L/∂w
 * b = b - α·∂L/∂b
 *
 * Where α is the learning rate.
 */

// Forward pass: y_pred = w * x + b
__global__ void linearRegressionForward(float *x, float *y_pred,
                                        float w, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        y_pred[idx] = w * x[idx] + b;
    }
}

// Compute gradients for weight and bias
__global__ void linearRegressionGradients(float *x, float *y_pred, float *y_true,
                                          float *grad_w, float *grad_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float error = y_pred[idx] - y_true[idx];
        
        // ∂L/∂w = (2/n) * error * x
        // ∂L/∂b = (2/n) * error
        atomicAdd(grad_w, (2.0f / n) * error * x[idx]);
        atomicAdd(grad_b, (2.0f / n) * error);
    }
}

// Compute Mean Squared Error loss
__global__ void computeMSELoss(float *y_pred, float *y_true, 
                               float *loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float error = y_pred[idx] - y_true[idx];
        atomicAdd(loss, error * error / n);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                 PART 2: LOGISTIC REGRESSION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Logistic Regression for binary classification (0 or 1).
 *
 * Architecture:
 * ────────────
 *        Input      Weighted Sum    Activation    Output
 *          x    →    z = w·x + b  →  σ(z)    →     ŷ
 *
 * Sigmoid Activation:
 * ──────────────────
 * σ(z) = 1 / (1 + e^(-z))
 *
 * Visual Sigmoid Function:
 * ───────────────────────
 *   1.0 ├─────────────────────
 *       │              ╱─────
 *   0.5 │         ╱───
 *       │    ╱────
 *   0.0 ├────────────────────
 *           -5   0   5
 *
 * Binary Cross-Entropy Loss:
 * ─────────────────────────
 * L = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
 *
 * Decision Boundary:
 * ─────────────────
 * Class 1 if ŷ ≥ 0.5
 * Class 0 if ŷ < 0.5
 *
 * Example Classification:
 * ──────────────────────
 *   Feature 2
 *   │  •  •  •   (Class 1)
 *   │     ╱
 *   │    ╱ Decision
 *   │   ╱  Boundary
 *   │  •  •  •   (Class 0)
 *   └───────────── Feature 1
 */

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Forward pass: y_pred = sigmoid(w * x + b)
__global__ void logisticRegressionForward(float *x, float *y_pred,
                                          float w, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float z = w * x[idx] + b;
        y_pred[idx] = sigmoid(z);
    }
}

// Binary cross-entropy loss
__global__ void binaryCrossEntropyLoss(float *y_pred, float *y_true,
                                       float *loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float pred = fmaxf(fminf(y_pred[idx], 0.9999f), 0.0001f);  // Clip for stability
        float l = -(y_true[idx] * logf(pred) + 
                    (1.0f - y_true[idx]) * logf(1.0f - pred));
        atomicAdd(loss, l / n);
    }
}

// Gradients for logistic regression
__global__ void logisticRegressionGradients(float *x, float *y_pred, float *y_true,
                                            float *grad_w, float *grad_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float error = y_pred[idx] - y_true[idx];
        
        // Gradient is same as linear regression due to sigmoid derivative!
        atomicAdd(grad_w, error * x[idx] / n);
        atomicAdd(grad_b, error / n);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *             PART 3: MULTI-CLASS CLASSIFICATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Softmax for classifying into multiple categories.
 *
 * Architecture:
 * ────────────
 *                Input Features
 *                 x₁  x₂  x₃
 *                 │   │   │
 *              ┌──┴───┴───┴──┐
 *              │   Weights   │  W·x + b
 *              └──┬───┬───┬──┘
 *                z₁  z₂  z₃
 *                 │   │   │
 *              ┌──┴───┴───┴──┐
 *              │   Softmax   │
 *              └──┬───┬───┬──┘
 *                p₁  p₂  p₃    Probabilities
 *
 * Softmax Function:
 * ────────────────
 * pᵢ = exp(zᵢ) / Σexp(zⱼ)
 *
 * Properties:
 * - All outputs are positive
 * - Sum of outputs = 1.0 (valid probability distribution)
 * - Argmax gives predicted class
 *
 * Example (3 classes):
 * ───────────────────
 * Logits:  [2.0, 1.0, 0.1]
 *          ↓ Softmax
 * Probs:   [0.659, 0.242, 0.099]
 *          ↓ Argmax
 * Class:   0 (highest probability)
 *
 * Cross-Entropy Loss:
 * ──────────────────
 * L = -Σ yᵢ·log(pᵢ)
 *
 * Where y is one-hot encoded true label.
 */

// Softmax activation (numerically stable version)
__global__ void softmaxForward(float *logits, float *probs, 
                               int n, int num_classes) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < n) {
        int offset = sample * num_classes;
        
        // Find max for numerical stability
        float max_logit = logits[offset];
        for (int i = 1; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, logits[offset + i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            probs[offset + i] = expf(logits[offset + i] - max_logit);
            sum += probs[offset + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            probs[offset + i] /= sum;
        }
    }
}

// Cross-entropy loss for multi-class classification
__global__ void crossEntropyLoss(float *probs, int *labels,
                                 float *loss, int n, int num_classes) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < n) {
        int label = labels[sample];
        float pred = fmaxf(probs[sample * num_classes + label], 1e-7f);
        atomicAdd(loss, -logf(pred) / n);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *           PART 4: FEEDFORWARD NEURAL NETWORK
 * ═══════════════════════════════════════════════════════════════════
 *
 * Multi-layer neural network with hidden layers.
 *
 * Architecture (2-layer network):
 * ──────────────────────────────
 *
 *   Input Layer    Hidden Layer    Output Layer
 *   ┌────┐         ┌────┐          ┌────┐
 *   │ x₁ │───┐     │ h₁ │───┐      │ y₁ │
 *   └────┘   │  ┌─→└────┘   │   ┌─→└────┘
 *            ├─→│           ├──→│
 *   ┌────┐   │  │  ┌────┐   │   │  ┌────┐
 *   │ x₂ │───┼─→├─→│ h₂ │───┼──→├─→│ y₂ │
 *   └────┘   │  │  └────┘   │   │  └────┘
 *            ├─→│           ├──→│
 *   ┌────┐   │  │  ┌────┐   │   │  ┌────┐
 *   │ x₃ │───┘  └─→│ h₃ │───┘   └─→│ y₃ │
 *   └────┘         └────┘          └────┘
 *
 *   Input        W₁, b₁        W₂, b₂      Output
 *   (n_in)      ReLU           Softmax     (n_out)
 *              (n_hidden)
 *
 * Forward Pass:
 * ────────────
 * h = ReLU(W₁·x + b₁)
 * y = Softmax(W₂·h + b₂)
 *
 * ReLU Activation:
 * ───────────────
 * ReLU(x) = max(0, x)
 *
 * Visual:
 *   Output
 *   │     ╱
 *   │   ╱
 *   │ ╱
 *   │╱____________ Input
 *   │  (negative values become 0)
 *
 * Backpropagation:
 * ───────────────
 * Computes gradients layer-by-layer using chain rule:
 * 
 * ∂L/∂W₂ = ∂L/∂y · ∂y/∂W₂
 * ∂L/∂W₁ = ∂L/∂y · ∂y/∂h · ∂h/∂W₁
 */

// ReLU activation
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// ReLU derivative (for backprop)
__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Matrix multiplication: C = A * B
// A is m×k, B is k×n, C is m×n
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Add bias and apply ReLU: y = ReLU(x + b)
__global__ void addBiasReLU(float *x, float *b, float *y, 
                            int batch_size, int features) {
    int sample = blockIdx.y * blockDim.y + threadIdx.y;
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample < batch_size && feature < features) {
        int idx = sample * features + feature;
        y[idx] = relu(x[idx] + b[feature]);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *      PART 5: CONVOLUTIONAL NEURAL NETWORK (CNN)
 * ═══════════════════════════════════════════════════════════════════
 *
 * CNN for image classification (e.g., digit recognition, object classification).
 *
 * Typical CNN Architecture:
 * ────────────────────────
 *
 *   Input Image          Conv Layer         Pooling          FC Layer
 *   ┌────────┐          ┌────────┐        ┌──────┐        ┌────┐
 *   │28×28×1 │  →       │28×28×32│   →    │14×14×32│  →   │ 10 │
 *   │        │ Conv+ReLU│        │ MaxPool│        │Flatten│    │
 *   └────────┘          └────────┘        └──────┘  →FC   └────┘
 *                        ↓ (3×3 filters)    (2×2)          Softmax
 *                    Extract features    Downsample      Classes
 *
 * Convolution Operation:
 * ─────────────────────
 * Applies filters/kernels to detect features (edges, textures, etc.)
 *
 * Input:        Filter:      Output:
 * ┌─────────┐   ┌───────┐   ┌─────────┐
 * │ 1 2 3 4 │   │ 1  0 │   │  ?  ?  ?│
 * │ 5 6 7 8 │ * │ 0 -1 │ = │  ?  ?  ?│
 * │ 9 0 1 2 │   └───────┘   └─────────┘
 * └─────────┘
 *
 * Convolution at position (i,j):
 * output[i][j] = Σ input[i+m][j+n] * filter[m][n]
 *
 * Max Pooling:
 * ───────────
 * Reduces spatial dimensions by taking maximum in each region.
 *
 * Input (4×4):       Max Pool (2×2):     Output (2×2):
 * ┌──────────────┐                       ┌──────┐
 * │ 1  3│ 2  4 │                        │ 6│ 8 │
 * │ 5  6│ 7  8 │    →   [6]  [8]    →  ├──┼───┤
 * ├─────┼──────┤        [9] [12]        │ 9│12 │
 * │ 2  9│ 1 12 │                        └──────┘
 * │ 0  1│ 3  4 │
 * └──────────────┘
 *
 * Complete CNN Data Flow:
 * ──────────────────────
 *
 * 1. INPUT:  [28×28×1] Raw pixel values
 *            ↓
 * 2. CONV1:  Apply 32 filters (3×3) → [28×28×32]
 *            Learn low-level features (edges, corners)
 *            ↓
 * 3. RELU:   Activation → [28×28×32]
 *            Add non-linearity
 *            ↓
 * 4. POOL1:  Max pool (2×2) → [14×14×32]
 *            Reduce spatial dimensions
 *            ↓
 * 5. CONV2:  Apply 64 filters (3×3) → [14×14×64]
 *            Learn higher-level features
 *            ↓
 * 6. RELU:   Activation → [14×14×64]
 *            ↓
 * 7. POOL2:  Max pool (2×2) → [7×7×64]
 *            ↓
 * 8. FLATTEN: → [3136×1]
 *            ↓
 * 9. FC:     Fully connected → [128]
 *            Learn global patterns
 *            ↓
 * 10. RELU:  Activation → [128]
 *            ↓
 * 11. FC:    Output layer → [10]
 *            ↓
 * 12. SOFTMAX: Class probabilities
 */

// 2D Convolution with single input channel
__global__ void conv2d(float *input, float *filters, float *output,
                       int input_h, int input_w,
                       int filter_h, int filter_w,
                       int num_filters) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_idx = blockIdx.z;
    
    if (out_col < input_w - filter_w + 1 && 
        out_row < input_h - filter_h + 1 &&
        filter_idx < num_filters) {
        
        float sum = 0.0f;
        
        // Apply filter
        for (int fh = 0; fh < filter_h; fh++) {
            for (int fw = 0; fw < filter_w; fw++) {
                int in_row = out_row + fh;
                int in_col = out_col + fw;
                
                int input_idx = in_row * input_w + in_col;
                int filter_offset = filter_idx * (filter_h * filter_w);
                int filter_local = fh * filter_w + fw;
                
                sum += input[input_idx] * filters[filter_offset + filter_local];
            }
        }
        
        // Write output
        int out_h = input_h - filter_h + 1;
        int out_w = input_w - filter_w + 1;
        int out_idx = filter_idx * (out_h * out_w) + out_row * out_w + out_col;
        output[out_idx] = relu(sum);  // Apply ReLU
    }
}

// Max pooling (2×2)
__global__ void maxPool2x2(float *input, float *output,
                           int input_h, int input_w, int channels) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;
    
    int out_h = input_h / 2;
    int out_w = input_w / 2;
    
    if (out_col < out_w && out_row < out_h && channel < channels) {
        int in_row = out_row * 2;
        int in_col = out_col * 2;
        
        // Find max in 2×2 window
        float max_val = -INFINITY;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int in_idx = channel * (input_h * input_w) + 
                            (in_row + i) * input_w + (in_col + j);
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        
        int out_idx = channel * (out_h * out_w) + out_row * out_w + out_col;
        output[out_idx] = max_val;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *           PART 6: OBJECT DETECTION BASICS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Simple object detection: localize and classify objects in images.
 *
 * Output Format:
 * ─────────────
 * For each detection:
 * [x, y, w, h, class_scores...]
 *
 * Where:
 * - (x, y): Bounding box center
 * - (w, h): Bounding box width and height
 * - class_scores: Probability for each class
 *
 * Visual Example:
 * ──────────────
 * Image:                Detection:
 * ┌───────────────┐     ┌───────────────┐
 * │               │     │               │
 * │   ┌─────┐     │     │   ┌─────┐     │
 * │   │ DOG │     │  →  │   │ DOG │ 95% │
 * │   └─────┘     │     │   └─────┘     │
 * │        ┌───┐  │     │        ┌───┐  │
 * │        │CAT│  │     │        │CAT│80%│
 * │        └───┘  │     │        └───┘  │
 * └───────────────┘     └───────────────┘
 *
 * Architecture:
 * ────────────
 *   Input Image
 *       ↓
 *   CNN Backbone (Feature Extraction)
 *       ↓
 *   ┌───────────┬────────────┐
 *   ↓           ↓            ↓
 * Class      BBox         Confidence
 * Prediction Regression   Score
 * [C classes] [x,y,w,h]   [0-1]
 *
 * Loss Function:
 * ─────────────
 * L = L_class + λ_bbox·L_bbox + λ_conf·L_conf
 *
 * Where:
 * - L_class: Classification loss (cross-entropy)
 * - L_bbox: Bounding box loss (MSE or IoU)
 * - L_conf: Confidence loss (BCE)
 */

// Intersection over Union (IoU) for bounding boxes
__device__ float computeIoU(float x1, float y1, float w1, float h1,
                            float x2, float y2, float w2, float h2) {
    // Convert to corner coordinates
    float left1 = x1 - w1 / 2, right1 = x1 + w1 / 2;
    float top1 = y1 - h1 / 2, bottom1 = y1 + h1 / 2;
    
    float left2 = x2 - w2 / 2, right2 = x2 + w2 / 2;
    float top2 = y2 - h2 / 2, bottom2 = y2 + h2 / 2;
    
    // Intersection area
    float inter_left = fmaxf(left1, left2);
    float inter_right = fminf(right1, right2);
    float inter_top = fmaxf(top1, top2);
    float inter_bottom = fminf(bottom1, bottom2);
    
    float inter_w = fmaxf(0.0f, inter_right - inter_left);
    float inter_h = fmaxf(0.0f, inter_bottom - inter_top);
    float inter_area = inter_w * inter_h;
    
    // Union area
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;
    
    return inter_area / (union_area + 1e-6f);
}

// Non-Maximum Suppression (NMS) to remove duplicate detections
__global__ void nms(float *boxes, float *scores, bool *keep,
                    int num_boxes, float iou_threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_boxes && keep[i]) {
        float x1 = boxes[i * 4 + 0];
        float y1 = boxes[i * 4 + 1];
        float w1 = boxes[i * 4 + 2];
        float h1 = boxes[i * 4 + 3];
        float score1 = scores[i];
        
        // Compare with all subsequent boxes
        for (int j = i + 1; j < num_boxes; j++) {
            if (keep[j]) {
                float x2 = boxes[j * 4 + 0];
                float y2 = boxes[j * 4 + 1];
                float w2 = boxes[j * 4 + 2];
                float h2 = boxes[j * 4 + 3];
                float score2 = scores[j];
                
                float iou = computeIoU(x1, y1, w1, h1, x2, y2, w2, h2);
                
                // Suppress box with lower score if IoU is high
                if (iou > iou_threshold && score1 > score2) {
                    keep[j] = false;
                }
            }
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     DATA GENERATION UTILITIES
 * ═══════════════════════════════════════════════════════════════════
 */

// Generate synthetic linear regression data: y = 2x + 1 + noise
void generateLinearData(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)i / n * 10.0f;  // x in [0, 10]
        y[i] = 2.0f * x[i] + 1.0f + ((rand() % 100) / 100.0f - 0.5f);
    }
}

// Generate synthetic classification data (2 classes)
void generateClassificationData(float *x, int *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)(rand() % 1000) / 1000.0f * 10.0f;
        
        // Simple decision boundary at x = 5
        y[i] = (x[i] > 5.0f) ? 1 : 0;
        
        // Add some noise to make it interesting
        if (rand() % 10 == 0) {
            y[i] = 1 - y[i];
        }
    }
}

// Generate synthetic MNIST-like data (simple patterns)
void generateMNISTLikeData(float *images, int *labels, int n) {
    int img_size = 28 * 28;
    
    for (int i = 0; i < n; i++) {
        int label = rand() % 10;
        labels[i] = label;
        
        // Create simple pattern based on label
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int idx = i * img_size + y * 28 + x;
                
                // Create diagonal lines, circles, etc based on label
                if (label == 0) {  // Circle
                    float dx = x - 14, dy = y - 14;
                    images[idx] = (sqrtf(dx*dx + dy*dy) < 8) ? 1.0f : 0.0f;
                } else if (label == 1) {  // Vertical line
                    images[idx] = (x >= 12 && x <= 15) ? 1.0f : 0.0f;
                } else {  // Random pattern for other digits
                    images[idx] = ((x + y * label) % 10 < 5) ? 1.0f : 0.0f;
                }
                
                // Add noise
                images[idx] += ((rand() % 100) / 1000.0f - 0.05f);
                images[idx] = fmaxf(0.0f, fminf(1.0f, images[idx]));
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
    printf("║      CUDA Tutorial: Deep Learning from Scratch       ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    srand(time(NULL));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 1: Linear Regression
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 1: Linear Regression\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int n_samples = 1000;
    float *h_x = (float*)malloc(n_samples * sizeof(float));
    float *h_y = (float*)malloc(n_samples * sizeof(float));
    
    generateLinearData(h_x, h_y, n_samples);
    
    printf("Generated %d samples: y = 2x + 1 + noise\n", n_samples);
    printf("Training linear regression to recover w≈2, b≈1\n\n");
    
    // Allocate device memory
    float *d_x, *d_y, *d_y_pred, *d_grad_w, *d_grad_b, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_x, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_pred, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n_samples * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n_samples * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize parameters
    float w = 0.1f, b = 0.1f;
    float learning_rate = 0.001f;
    int epochs = 100;
    
    int blockSize = 256;
    int gridSize = (n_samples + blockSize - 1) / blockSize;
    
    printf("Training for %d epochs with learning rate %.4f\n\n", 
           epochs, learning_rate);
    printf("Epoch    Loss      w        b\n");
    printf("─────────────────────────────────\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Zero gradients
        CUDA_CHECK(cudaMemset(d_grad_w, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        // Forward pass
        linearRegressionForward<<<gridSize, blockSize>>>(d_x, d_y_pred, 
                                                          w, b, n_samples);
        
        // Compute loss
        computeMSELoss<<<gridSize, blockSize>>>(d_y_pred, d_y, d_loss, n_samples);
        
        // Compute gradients
        linearRegressionGradients<<<gridSize, blockSize>>>(d_x, d_y_pred, d_y,
                                                            d_grad_w, d_grad_b,
                                                            n_samples);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get gradients
        float grad_w, grad_b, loss;
        CUDA_CHECK(cudaMemcpy(&grad_w, d_grad_w, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Update parameters
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;
        
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            printf("%5d    %.4f    %.4f   %.4f\n", epoch, loss, w, b);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float training_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&training_time, start, stop));
    
    printf("\nTraining completed in %.3f ms\n", training_time);
    printf("Final model: ŷ = %.4f·x + %.4f\n", w, b);
    printf("Target model: y = 2.0000·x + 1.0000\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_y_pred));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_grad_b));
    CUDA_CHECK(cudaFree(d_loss));
    free(h_x);
    free(h_y);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 2: Binary Classification (Logistic Regression)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 2: Binary Classification\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int *h_labels = (int*)malloc(n_samples * sizeof(int));
    h_x = (float*)malloc(n_samples * sizeof(float));
    float *h_y_binary = (float*)malloc(n_samples * sizeof(float));
    
    generateClassificationData(h_x, h_labels, n_samples);
    
    // Convert labels to float for loss computation
    for (int i = 0; i < n_samples; i++) {
        h_y_binary[i] = (float)h_labels[i];
    }
    
    printf("Generated %d samples with decision boundary at x=5\n", n_samples);
    printf("Training logistic regression classifier\n\n");
    
    // Allocate device memory
    float *d_x_cls, *d_y_cls, *d_y_pred_cls;
    CUDA_CHECK(cudaMalloc(&d_x_cls, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_cls, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_pred_cls, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_x_cls, h_x, n_samples * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_cls, h_y_binary, n_samples * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize parameters
    w = 0.0f;
    b = 0.0f;
    learning_rate = 0.01f;
    epochs = 200;
    
    printf("Training for %d epochs\n\n", epochs);
    printf("Epoch    Loss      Accuracy\n");
    printf("───────────────────────────\n");
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        CUDA_CHECK(cudaMemset(d_grad_w, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        // Forward pass
        logisticRegressionForward<<<gridSize, blockSize>>>(d_x_cls, d_y_pred_cls,
                                                            w, b, n_samples);
        
        // Compute loss
        binaryCrossEntropyLoss<<<gridSize, blockSize>>>(d_y_pred_cls, d_y_cls,
                                                         d_loss, n_samples);
        
        // Compute gradients
        logisticRegressionGradients<<<gridSize, blockSize>>>(d_x_cls, d_y_pred_cls,
                                                              d_y_cls, d_grad_w,
                                                              d_grad_b, n_samples);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Update parameters
        float grad_w, grad_b, loss;
        CUDA_CHECK(cudaMemcpy(&grad_w, d_grad_w, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;
        
        // Compute accuracy every 20 epochs
        if (epoch % 20 == 0 || epoch == epochs - 1) {
            float *h_pred = (float*)malloc(n_samples * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_pred, d_y_pred_cls, 
                                 n_samples * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            
            int correct = 0;
            for (int i = 0; i < n_samples; i++) {
                int pred_class = (h_pred[i] >= 0.5f) ? 1 : 0;
                if (pred_class == h_labels[i]) correct++;
            }
            float accuracy = 100.0f * correct / n_samples;
            
            printf("%5d    %.4f    %.2f%%\n", epoch, loss, accuracy);
            free(h_pred);
        }
    }
    
    printf("\nTraining completed!\n");
    printf("Final model: P(y=1) = σ(%.4f·x + %.4f)\n\n", w, b);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_x_cls));
    CUDA_CHECK(cudaFree(d_y_cls));
    CUDA_CHECK(cudaFree(d_y_pred_cls));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_grad_b));
    CUDA_CHECK(cudaFree(d_loss));
    free(h_x);
    free(h_y_binary);
    free(h_labels);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 3: CNN for Image Classification
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 3: CNN for Image Classification\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int num_images = 100;
    int img_height = 28, img_width = 28;
    int num_classes = 10;
    
    printf("Architecture:\n");
    printf("  Input:  28×28×1 grayscale images\n");
    printf("  Conv1:  3×3×8 filters → 26×26×8\n");
    printf("  Pool1:  2×2 max pool → 13×13×8\n");
    printf("  Output: 10 classes (digits 0-9)\n\n");
    
    // Generate synthetic data
    float *h_images = (float*)malloc(num_images * img_height * img_width * 
                                     sizeof(float));
    int *h_img_labels = (int*)malloc(num_images * sizeof(int));
    
    generateMNISTLikeData(h_images, h_img_labels, num_images);
    
    printf("Generated %d synthetic 28×28 images\n", num_images);
    printf("(In practice, use real MNIST dataset)\n\n");
    
    // Allocate device memory for one image (demo)
    float *d_image, *d_conv_out, *d_pool_out;
    int conv_h = 26, conv_w = 26, num_filters = 8;
    int pool_h = 13, pool_w = 13;
    
    CUDA_CHECK(cudaMalloc(&d_image, img_height * img_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_out, conv_h * conv_w * num_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool_out, pool_h * pool_w * num_filters * sizeof(float)));
    
    // Initialize filters (normally learned during training)
    float *h_filters = (float*)malloc(num_filters * 3 * 3 * sizeof(float));
    for (int i = 0; i < num_filters * 3 * 3; i++) {
        h_filters[i] = ((rand() % 100) / 100.0f - 0.5f) * 0.1f;
    }
    
    float *d_filters;
    CUDA_CHECK(cudaMalloc(&d_filters, num_filters * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filters, h_filters, 
                         num_filters * 3 * 3 * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Process first image as demo
    CUDA_CHECK(cudaMemcpy(d_image, h_images, 
                         img_height * img_width * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    printf("Forward pass through CNN:\n");
    
    // Conv layer
    dim3 convBlock(16, 16);
    dim3 convGrid((conv_w + 15) / 16, (conv_h + 15) / 16, num_filters);
    
    CUDA_CHECK(cudaEventRecord(start));
    conv2d<<<convGrid, convBlock>>>(d_image, d_filters, d_conv_out,
                                     img_height, img_width, 3, 3, num_filters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float conv_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&conv_time, start, stop));
    printf("  Conv2D (3×3×8): %.3f ms\n", conv_time);
    
    // Max pooling
    dim3 poolBlock(16, 16);
    dim3 poolGrid((pool_w + 15) / 16, (pool_h + 15) / 16, num_filters);
    
    CUDA_CHECK(cudaEventRecord(start));
    maxPool2x2<<<poolGrid, poolBlock>>>(d_conv_out, d_pool_out,
                                         conv_h, conv_w, num_filters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float pool_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pool_time, start, stop));
    printf("  MaxPool (2×2): %.3f ms\n", pool_time);
    printf("  Total forward pass: %.3f ms\n\n", conv_time + pool_time);
    
    printf("For complete training:\n");
    printf("  1. Add fully connected layers after pooling\n");
    printf("  2. Implement backpropagation for conv and pool layers\n");
    printf("  3. Train on real dataset (MNIST, CIFAR-10, etc.)\n");
    printf("  4. Use data augmentation for better generalization\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_conv_out));
    CUDA_CHECK(cudaFree(d_pool_out));
    CUDA_CHECK(cudaFree(d_filters));
    free(h_images);
    free(h_img_labels);
    free(h_filters);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. Linear regression: Simple gradient descent        ║\n");
    printf("║ 2. Classification: Sigmoid/softmax activations       ║\n");
    printf("║ 3. Neural networks: Layer-by-layer forward/backward  ║\n");
    printf("║ 4. CNNs: Convolution + pooling for spatial features  ║\n");
    printf("║ 5. GPU acceleration critical for deep learning       ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    printf("Next Steps:\n");
    printf("  • Implement full backpropagation\n");
    printf("  • Add batch normalization\n");
    printf("  • Implement dropout for regularization\n");
    printf("  • Use real datasets (MNIST, ImageNet)\n");
    printf("  • Explore modern architectures (ResNet, Transformers)\n");
    printf("  • Consider using cuDNN for production code\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. BASIC:
 *    - Implement polynomial regression (quadratic, cubic)
 *    - Add momentum to gradient descent
 *    - Implement mini-batch training
 *
 * 2. INTERMEDIATE:
 *    - Complete backpropagation for neural network
 *    - Implement Adam optimizer
 *    - Add dropout regularization
 *    - Implement batch normalization
 *
 * 3. ADVANCED:
 *    - Implement ResNet building blocks
 *    - Add data augmentation pipeline
 *    - Implement attention mechanism
 *    - Build YOLO-style object detector
 *
 * 4. OPTIMIZATION:
 *    - Use tensor cores for matrix multiplication
 *    - Implement mixed-precision training
 *    - Add gradient checkpointing for memory savings
 *    - Profile and optimize memory transfers
 *
 * ═══════════════════════════════════════════════════════════════════
 *                      RECOMMENDED RESOURCES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Libraries:
 * - cuDNN: NVIDIA's optimized deep learning primitives
 * - cuBLAS: Optimized matrix operations
 * - Thrust: High-level C++ parallel algorithms
 *
 * Datasets:
 * - MNIST: Handwritten digits (28×28 grayscale)
 * - CIFAR-10: Small color images (32×32 RGB)
 * - ImageNet: Large-scale image classification
 * - COCO: Object detection and segmentation
 *
 * Papers:
 * - AlexNet (2012): Deep CNN for ImageNet
 * - ResNet (2015): Residual connections
 * - Attention Is All You Need (2017): Transformers
 * - YOLO (2016): Real-time object detection
 *
 * ═══════════════════════════════════════════════════════════════════
 */

