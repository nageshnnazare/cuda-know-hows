# CUDA Thread Indexing Patterns: Complete Guide
## 1D, 2D, and 3D Thread Access with Detailed Illustrations

This tutorial provides an in-depth exploration of thread indexing patterns in CUDA, from basic 1D arrays to complex 3D volumes, with visual illustrations and practical examples.

---

## Table of Contents

1. [Understanding Thread Hierarchy](#understanding-thread-hierarchy)
2. [1D Thread Indexing](#1d-thread-indexing)
3. [2D Thread Indexing](#2d-thread-indexing)
4. [3D Thread Indexing](#3d-thread-indexing)
5. [Advanced Patterns](#advanced-patterns)
6. [Common Pitfalls](#common-pitfalls)
7. [Performance Considerations](#performance-considerations)

---

## Understanding Thread Hierarchy

### CUDA Thread Organization

```
┌──────────────────────────────────────────────────────────────────┐
│                    CUDA THREAD HIERARCHY                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Level 1: GRID (Entire Kernel Launch)                            │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                                                         │      │
│  │  Level 2: BLOCKS (Thread Groups)                       │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │      │
│  │  │ Block    │  │ Block    │  │ Block    │            │      │
│  │  │ (0,0,0)  │  │ (1,0,0)  │  │ (2,0,0)  │            │      │
│  │  │          │  │          │  │          │            │      │
│  │  │  Level 3: THREADS (Individual Execution Units)     │      │
│  │  │  ┌─┬─┬─┐ │  │  ┌─┬─┬─┐ │  │  ┌─┬─┬─┐ │            │      │
│  │  │  │T│T│T│ │  │  │T│T│T│ │  │  │T│T│T│ │            │      │
│  │  │  ├─┼─┼─┤ │  │  ├─┼─┼─┤ │  │  ├─┼─┼─┤ │            │      │
│  │  │  │T│T│T│ │  │  │T│T│T│ │  │  │T│T│T│ │            │      │
│  │  │  └─┴─┴─┘ │  │  └─┴─┴─┘ │  │  └─┴─┴─┘ │            │      │
│  │  └──────────┘  └──────────┘  └──────────┘            │      │
│  │                                                         │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  Built-in Variables:                                              │
│  ────────────────────                                             │
│  • gridDim.{x,y,z}   - Number of blocks in each dimension        │
│  • blockIdx.{x,y,z}  - Current block index                       │
│  • blockDim.{x,y,z}  - Number of threads per block               │
│  • threadIdx.{x,y,z} - Current thread index within block         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key Concepts

**Grid**: Collection of all blocks launched by a kernel
- Can be 1D, 2D, or 3D
- Defined by `gridDim`

**Block**: Group of threads that can cooperate
- Can be 1D, 2D, or 3D
- Defined by `blockDim`
- Maximum: 1024 threads per block (architecture dependent)

**Thread**: Individual execution unit
- Has unique ID within block and globally
- Executes kernel code

---

## 1D Thread Indexing

### Basic 1D Configuration

```
┌──────────────────────────────────────────────────────────────────┐
│                  1D THREAD ORGANIZATION                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Grid: 4 blocks (gridDim.x = 4)                                  │
│  Block: 8 threads (blockDim.x = 8)                               │
│  Total: 32 threads                                               │
│                                                                   │
│  ┌──────────┬──────────┬──────────┬──────────┐                  │
│  │ Block 0  │ Block 1  │ Block 2  │ Block 3  │                  │
│  │ (bIdx=0) │ (bIdx=1) │ (bIdx=2) │ (bIdx=3) │                  │
│  ├──────────┼──────────┼──────────┼──────────┤                  │
│  │┌─┬─┬─┬─┐ │┌─┬─┬─┬─┐ │┌─┬─┬─┬─┐ │┌─┬─┬─┬─┐ │                  │
│  ││0│1│2│3│ ││0│1│2│3│ ││0│1│2│3│ ││0│1│2│3│ │                  │
│  │├─┼─┼─┼─┤ │├─┼─┼─┼─┤ │├─┼─┼─┼─┤ │├─┼─┼─┼─┤ │                  │
│  ││4│5│6│7│ ││4│5│6│7│ ││4│5│6│7│ ││4│5│6│7│ │                  │
│  │└─┴─┴─┴─┘ │└─┴─┴─┴─┘ │└─┴─┴─┴─┘ │└─┴─┴─┴─┘ │                  │
│  │  tIdx    │  tIdx    │  tIdx    │  tIdx    │                  │
│  └──────────┴──────────┴──────────┴──────────┘                  │
│                                                                   │
│  Global Thread ID Calculation:                                   │
│  ────────────────────────────────                                │
│  globalID = blockIdx.x * blockDim.x + threadIdx.x                │
│                                                                   │
│  Examples:                                                        │
│  • Block 0, Thread 0: 0 * 8 + 0 = 0                              │
│  • Block 0, Thread 7: 0 * 8 + 7 = 7                              │
│  • Block 1, Thread 0: 1 * 8 + 0 = 8                              │
│  • Block 1, Thread 5: 1 * 8 + 5 = 13                             │
│  • Block 3, Thread 7: 3 * 8 + 7 = 31                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1D Example: Vector Addition

```
┌──────────────────────────────────────────────────────────────────┐
│              1D INDEXING: VECTOR ADDITION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Problem: Add two vectors of size N                              │
│  C[i] = A[i] + B[i], for i = 0 to N-1                           │
│                                                                   │
│  Data Layout (N = 16):                                            │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐            │
│  │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│  Indices   │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤            │
│  │a0│a1│a2│a3│a4│a5│a6│a7│a8│a9│..│..│..│..│..│..│  Array A   │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤            │
│  │b0│b1│b2│b3│b4│b5│b6│b7│b8│b9│..│..│..│..│..│..│  Array B   │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘            │
│                                                                   │
│  Thread Mapping (4 blocks × 4 threads = 16 threads):             │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Block 0          Block 1          Block 2    Block 3  │      │
│  │  ┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐   ┌──┬──┐   ┌──┬──┐  │      │
│  │  │T0│T1│T2│T3│   │T4│T5│T6│T7│   │T8│T9│   │..|..│  │      │
│  │  └┬─┴┬─┴┬─┴┬─┘   └┬─┴┬─┴┬─┴┬─┘   └┬─┴┬─┘   └┬─┴┬─┘  │      │
│  │   ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓      ↓  ↓      ↓  ↓    │      │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  │      │
│  │  │c0│c1│c2│c3│c4│c5│c6│c7│c8│c9│10│11│12│13│14│15│  │      │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  │      │
│  │                     Result Array C                     │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  Each thread processes ONE element:                               │
│  Thread 0  → C[0]  = A[0]  + B[0]                                │
│  Thread 5  → C[5]  = A[5]  + B[5]                                │
│  Thread 15 → C[15] = A[15] + B[15]                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void vectorAdd1D(float *A, float *B, float *C, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch configuration
int N = 1024;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### 1D Strided Access Pattern

```
┌──────────────────────────────────────────────────────────────────┐
│              1D STRIDED ACCESS PATTERN                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Use Case: Processing with grid-stride loop                      │
│  (When N > total number of threads)                              │
│                                                                   │
│  Array (N = 20):                                                  │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ │
│  │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│16│17│18│19│ │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ │
│   ↑           ↑           ↑           ↑           ↑              │
│   T0          T0          T0          T0          T0             │
│   (1st)       (2nd)       (3rd)       (4th)       (5th)          │
│                                                                   │
│  With only 4 threads total (stride = 4):                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Thread 0 processes: 0, 4, 8, 12, 16                 │        │
│  │  Thread 1 processes: 1, 5, 9, 13, 17                 │        │
│  │  Thread 2 processes: 2, 6, 10, 14, 18                │        │
│  │  Thread 3 processes: 3, 7, 11, 15, 19                │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void vectorProcessStride(float *data, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate stride (total number of threads)
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}

// Launch with limited threads
int threadsPerBlock = 256;
int blocksPerGrid = 4;  // Only 1024 threads total

vectorProcessStride<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
```

---

## 2D Thread Indexing

### Basic 2D Configuration

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      2D THREAD ORGANIZATION                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Grid: 3×2 blocks (gridDim.x=3, gridDim.y=2)                            │
│  Block: 4×4 threads (blockDim.x=4, blockDim.y=4)                        │
│  Total: 6 blocks × 16 threads = 96 threads                              │
│                                                                           │
│  GRID LAYOUT:                                                             │
│  ┌─────────────────┬─────────────────┬─────────────────┐                │
│  │                 │                 │                 │                │
│  │   Block(0,1)    │   Block(1,1)    │   Block(2,1)    │  ← y=1        │
│  │   bIdx.x=0      │   bIdx.x=1      │   bIdx.x=2      │                │
│  │   bIdx.y=1      │   bIdx.y=1      │   bIdx.y=1      │                │
│  │                 │                 │                 │                │
│  ├─────────────────┼─────────────────┼─────────────────┤                │
│  │                 │                 │                 │                │
│  │   Block(0,0)    │   Block(1,0)    │   Block(2,0)    │  ← y=0        │
│  │   bIdx.x=0      │   bIdx.x=1      │   bIdx.x=2      │                │
│  │   bIdx.y=0      │   bIdx.y=0      │   bIdx.y=0      │                │
│  │                 │                 │                 │                │
│  └─────────────────┴─────────────────┴─────────────────┘                │
│          ↑                 ↑                 ↑                            │
│        x=0               x=1               x=2                           │
│                                                                           │
│  BLOCK LAYOUT (Each block contains 4×4 threads):                         │
│  ┌────────────────────────────────────────────────────┐                  │
│  │                    y-axis                           │                  │
│  │                      ↑                              │                  │
│  │    ┌────┬────┬────┬────┐                           │                  │
│  │  3 │ 03 │ 13 │ 23 │ 33 │  threadIdx.y              │                  │
│  │    ├────┼────┼────┼────┤                           │                  │
│  │  2 │ 02 │ 12 │ 22 │ 32 │                           │                  │
│  │    ├────┼────┼────┼────┤                           │                  │
│  │  1 │ 01 │ 11 │ 21 │ 31 │                           │                  │
│  │    ├────┼────┼────┼────┤                           │                  │
│  │  0 │ 00 │ 10 │ 20 │ 30 │                           │                  │
│  │    └────┴────┴────┴────┘                           │                  │
│  │         0    1    2    3  → x-axis                 │                  │
│  │              threadIdx.x                            │                  │
│  │                                                     │                  │
│  │  Thread(1,2) means: threadIdx.x=1, threadIdx.y=2   │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  GLOBAL POSITION CALCULATION:                                             │
│  ────────────────────────────                                             │
│  col (x) = blockIdx.x * blockDim.x + threadIdx.x                         │
│  row (y) = blockIdx.y * blockDim.y + threadIdx.y                         │
│                                                                           │
│  Example: Block(1,0), Thread(2,3)                                        │
│  • col = 1 * 4 + 2 = 6                                                   │
│  • row = 0 * 4 + 3 = 3                                                   │
│  • Global position: (6, 3)                                               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2D Example: Image Processing

```
┌──────────────────────────────────────────────────────────────────────────┐
│               2D INDEXING: IMAGE GRAYSCALE CONVERSION                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Problem: Convert RGB image to grayscale                                 │
│  Image: 12×8 pixels (width × height)                                     │
│                                                                           │
│  RGB IMAGE (Each pixel has 3 channels: R, G, B):                         │
│  ┌────────────────────────────────────────────────────┐                  │
│  │ [R,G,B] [R,G,B] [R,G,B] ... [R,G,B]  ← Row 7       │                  │
│  │ [R,G,B] [R,G,B] [R,G,B] ... [R,G,B]  ← Row 6       │                  │
│  │ [R,G,B] [R,G,B] [R,G,B] ... [R,G,B]  ← Row 5       │                  │
│  │   ...     ...     ...   ...   ...                   │                  │
│  │ [R,G,B] [R,G,B] [R,G,B] ... [R,G,B]  ← Row 1       │                  │
│  │ [R,G,B] [R,G,B] [R,G,B] ... [R,G,B]  ← Row 0       │                  │
│  │   ↑       ↑       ↑           ↑                     │                  │
│  │  Col0    Col1    Col2        Col11                  │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  THREAD MAPPING (3×2 blocks, 4×4 threads/block):                         │
│  ┌────────────────────────────────────────────────────┐                  │
│  │           Block(0,1)  Block(1,1)  Block(2,1)       │                  │
│  │             ┌────┐      ┌────┐      ┌────┐         │                  │
│  │  Rows 4-7   │4×4 │      │4×4 │      │4×4 │         │                  │
│  │             └────┘      └────┘      └────┘         │                  │
│  │                                                     │                  │
│  │           Block(0,0)  Block(1,0)  Block(2,0)       │                  │
│  │             ┌────┐      ┌────┐      ┌────┐         │                  │
│  │  Rows 0-3   │4×4 │      │4×4 │      │4×4 │         │                  │
│  │             └────┘      └────┘      └────┘         │                  │
│  │             Cols       Cols        Cols             │                  │
│  │             0-3        4-7         8-11             │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  MEMORY ACCESS PATTERN:                                                   │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Thread at (col=5, row=3):                         │                  │
│  │                                                     │                  │
│  │  1. Calculate linear index:                        │                  │
│  │     idx = row * width + col                        │                  │
│  │         = 3 * 12 + 5 = 41                          │                  │
│  │                                                     │                  │
│  │  2. Access RGB data:                                │                  │
│  │     rgbIdx = idx * 3 = 41 * 3 = 123                │                  │
│  │     R = rgb[123]                                    │                  │
│  │     G = rgb[124]                                    │                  │
│  │     B = rgb[125]                                    │                  │
│  │                                                     │                  │
│  │  3. Calculate grayscale:                            │                  │
│  │     gray = 0.299*R + 0.587*G + 0.114*B             │                  │
│  │                                                     │                  │
│  │  4. Write to output:                                │                  │
│  │     grayImage[41] = gray                            │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void rgbToGrayscale2D(unsigned char *rgb, 
                                  unsigned char *gray,
                                  int width, int height) {
    // Calculate 2D position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (col < width && row < height) {
        // Convert 2D to 1D index
        int grayIdx = row * width + col;
        int rgbIdx = grayIdx * 3;
        
        // Get RGB values
        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        
        // Convert to grayscale
        gray[grayIdx] = (unsigned char)(0.299f * r + 
                                        0.587f * g + 
                                        0.114f * b);
    }
}

// Launch configuration
dim3 blockSize(16, 16);  // 16×16 = 256 threads
dim3 gridSize((width + 15) / 16, (height + 15) / 16);

rgbToGrayscale2D<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);
```

### 2D Example: Matrix Transpose

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   2D INDEXING: MATRIX TRANSPOSE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Problem: Transpose a matrix A → A^T                                     │
│  A[i][j] → A^T[j][i]                                                     │
│                                                                           │
│  INPUT MATRIX A (8×8):                                                    │
│  ┌──────────────────────────────────────────────┐                        │
│  │  Col: 0   1   2   3   4   5   6   7          │                        │
│  │     ┌───┬───┬───┬───┬───┬───┬───┬───┐       │                        │
│  │ R 0 │ 00│ 01│ 02│ 03│ 04│ 05│ 06│ 07│       │                        │
│  │ o   ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │ w 1 │ 10│ 11│ 12│ 13│ 14│ 15│ 16│ 17│       │                        │
│  │ :   ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   2 │ 20│ 21│ 22│ 23│ 24│ 25│ 26│ 27│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   3 │ 30│ 31│ 32│ 33│ 34│ 35│ 36│ 37│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   4 │ 40│ 41│ 42│ 43│ 44│ 45│ 46│ 47│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   5 │ 50│ 51│ 52│ 53│ 54│ 55│ 56│ 57│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   6 │ 60│ 61│ 62│ 63│ 64│ 65│ 66│ 67│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   7 │ 70│ 71│ 72│ 73│ 74│ 75│ 76│ 77│       │                        │
│  │     └───┴───┴───┴───┴───┴───┴───┴───┘       │                        │
│  └──────────────────────────────────────────────┘                        │
│                                                                           │
│  OUTPUT MATRIX A^T (8×8):                                                 │
│  ┌──────────────────────────────────────────────┐                        │
│  │  Col: 0   1   2   3   4   5   6   7          │                        │
│  │     ┌───┬───┬───┬───┬───┬───┬───┬───┐       │                        │
│  │ R 0 │ 00│ 10│ 20│ 30│ 40│ 50│ 60│ 70│       │                        │
│  │ o   ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │ w 1 │ 01│ 11│ 21│ 31│ 41│ 51│ 61│ 71│       │                        │
│  │ :   ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   2 │ 02│ 12│ 22│ 32│ 42│ 52│ 62│ 72│       │                        │
│  │     ├───┼───┼───┼───┼───┼───┼───┼───┤       │                        │
│  │   3 │ 03│ 13│ 23│ 33│ 43│ 53│ 63│ 73│       │                        │
│  │     └───┴───┴───┴───┴───┴───┴───┴───┘       │                        │
│  │          (Rows and columns swapped)          │                        │
│  └──────────────────────────────────────────────┘                        │
│                                                                           │
│  THREAD PROCESSING:                                                       │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Thread at (col=2, row=5):                         │                  │
│  │                                                     │                  │
│  │  Input:  A[5][2] at index (5 * 8 + 2) = 42        │                  │
│  │  Output: A^T[2][5] at index (2 * 8 + 5) = 21      │                  │
│  │                                                     │                  │
│  │  Read from:  input[row * width + col]              │                  │
│  │  Write to:   output[col * height + row]            │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void transposeNaive(float *input, float *output,
                               int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Read from input[row][col]
        int inputIdx = row * width + col;
        
        // Write to output[col][row]
        int outputIdx = col * height + row;
        
        output[outputIdx] = input[inputIdx];
    }
}

// Launch
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);

transposeNaive<<<gridSize, blockSize>>>(d_in, d_out, width, height);
```

---

## 3D Thread Indexing

### Basic 3D Configuration

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      3D THREAD ORGANIZATION                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Grid: 2×2×2 blocks (gridDim: x=2, y=2, z=2)                            │
│  Block: 4×4×4 threads (blockDim: x=4, y=4, z=4)                         │
│  Total: 8 blocks × 64 threads = 512 threads                             │
│                                                                           │
│  GRID LAYOUT (3D visualization):                                          │
│                                                                           │
│         Z=1 Layer (Top)                                                   │
│         ┌───────────────────┬───────────────────┐                        │
│        │                   │                   │                        │
│        │   Block(0,1,1)    │   Block(1,1,1)    │                        │
│        │                   │                   │                        │
│        ├───────────────────┼───────────────────┤                        │
│        │                   │                   │                        │
│        │   Block(0,0,1)    │   Block(1,0,1)    │                        │
│        │                   │                   │                        │
│        └───────────────────┴───────────────────┘                        │
│                                                                           │
│         Z=0 Layer (Bottom)                                                │
│         ┌───────────────────┬───────────────────┐                        │
│        │                   │                   │                        │
│        │   Block(0,1,0)    │   Block(1,1,0)    │                        │
│        │                   │                   │                        │
│        ├───────────────────┼───────────────────┤                        │
│        │                   │                   │                        │
│        │   Block(0,0,0)    │   Block(1,0,0)    │                        │
│        │                   │                   │                        │
│        └───────────────────┴───────────────────┘                        │
│                                                                           │
│  BLOCK LAYOUT (4×4×4 cube of threads):                                   │
│  ┌────────────────────────────────────────────────────┐                  │
│  │                                                     │                  │
│  │    Z=3  ┌──┬──┬──┬──┐  (Top slice)                │                  │
│  │         │  │  │  │  │                              │                  │
│  │         └──┴──┴──┴──┘                              │                  │
│  │                                                     │                  │
│  │    Z=2  ┌──┬──┬──┬──┐                              │                  │
│  │         │  │  │  │  │                              │                  │
│  │         └──┴──┴──┴──┘                              │                  │
│  │                                                     │                  │
│  │    Z=1  ┌──┬──┬──┬──┐                              │                  │
│  │         │  │  │  │  │                              │                  │
│  │         └──┴──┴──┴──┘                              │                  │
│  │                                                     │                  │
│  │    Z=0  ┌──┬──┬──┬──┐  (Bottom slice)             │                  │
│  │         │00│10│20│30│                              │                  │
│  │         └──┴──┴──┴──┘                              │                  │
│  │         X: 0  1  2  3                              │                  │
│  │         (4 threads along X-axis per slice)         │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  GLOBAL POSITION CALCULATION:                                             │
│  ────────────────────────────                                             │
│  x = blockIdx.x * blockDim.x + threadIdx.x                               │
│  y = blockIdx.y * blockDim.y + threadIdx.y                               │
│  z = blockIdx.z * blockDim.z + threadIdx.z                               │
│                                                                           │
│  LINEAR INDEX (for 1D array access):                                     │
│  index = z * (width * height) + y * width + x                            │
│                                                                           │
│  Example: Block(1,0,1), Thread(2,3,1)                                    │
│  • x = 1 * 4 + 2 = 6                                                     │
│  • y = 0 * 4 + 3 = 3                                                     │
│  • z = 1 * 4 + 1 = 5                                                     │
│  • Position: (6, 3, 5) in volume                                         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3D Example: Volume Processing

```
┌──────────────────────────────────────────────────────────────────────────┐
│                3D INDEXING: MEDICAL VOLUME PROCESSING                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Problem: Apply 3D filter to medical CT/MRI volume                       │
│  Volume: 256×256×128 (width × height × depth)                            │
│                                                                           │
│  3D VOLUME VISUALIZATION:                                                 │
│  ┌────────────────────────────────────────────────────┐                  │
│  │                                                     │                  │
│  │  Depth (Z-axis)                                     │                  │
│  │    ↗                                                │                  │
│  │   /                                                 │                  │
│  │  /    Slice 127 (top)                              │                  │
│  │ /     ┌──────────────┐                             │                  │
│  │      │              │                             │                  │
│  │      │   Slice 64   │                             │                  │
│  │      │  ┌──────────┐│                             │                  │
│  │      │  │          ││                             │                  │
│  │      │  │ Slice 32 ││                             │                  │
│  │      │  │ ┌────────┤│                             │                  │
│  │      │  │ │  Slice││                             │                  │
│  │      │  │ │  0     ││ (bottom)                    │                  │
│  │      │  │ └────────┤│                             │                  │
│  │      │  └──────────┘│                             │                  │
│  │      └──────────────┘                             │                  │
│  │          │                                         │                  │
│  │          └─→ Width (X-axis)                       │                  │
│  │          ↓                                         │                  │
│  │     Height (Y-axis)                                │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  THREAD MAPPING (8×8×8 threads per block):                               │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Each block processes an 8×8×8 cube:               │                  │
│  │                                                     │                  │
│  │  Grid dimensions:                                   │                  │
│  │  • X: (256 + 7) / 8 = 32 blocks                    │                  │
│  │  • Y: (256 + 7) / 8 = 32 blocks                    │                  │
│  │  • Z: (128 + 7) / 8 = 16 blocks                    │                  │
│  │  Total: 32 × 32 × 16 = 16,384 blocks               │                  │
│  │                                                     │                  │
│  │  Each block has 8×8×8 = 512 threads                │                  │
│  │  Total threads: 16,384 × 512 = 8,388,608 threads   │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  MEMORY ACCESS:                                                           │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Thread at (x=100, y=150, z=64):                   │                  │
│  │                                                     │                  │
│  │  1. Calculate linear index:                        │                  │
│  │     idx = z * (width * height) + y * width + x     │                  │
│  │         = 64 * (256 * 256) + 150 * 256 + 100       │                  │
│  │         = 64 * 65536 + 38400 + 100                 │                  │
│  │         = 4,234,304 + 38,400 + 100                 │                  │
│  │         = 4,272,804                                 │                  │
│  │                                                     │                  │
│  │  2. Access voxel:                                   │                  │
│  │     value = volume[idx]                             │                  │
│  │                                                     │                  │
│  │  3. Process (e.g., 3D Gaussian filter):            │                  │
│  │     result = 0                                      │                  │
│  │     for dz in [-1, 0, 1]:                          │                  │
│  │       for dy in [-1, 0, 1]:                        │                  │
│  │         for dx in [-1, 0, 1]:                      │                  │
│  │           neighborIdx = (z+dz)*(W*H) +              │                  │
│  │                        (y+dy)*W + (x+dx)            │                  │
│  │           result += volume[neighborIdx] * weight    │                  │
│  │                                                     │                  │
│  │  4. Write result:                                   │                  │
│  │     output[idx] = result                            │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void process3DVolume(float *input, float *output,
                                int width, int height, int depth) {
    // Calculate 3D position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds
    if (x < width && y < height && z < depth) {
        // Calculate linear index
        int idx = z * (width * height) + y * width + x;
        
        // Apply 3D filter (example: simple averaging)
        float sum = 0.0f;
        int count = 0;
        
        // 3×3×3 neighborhood
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    
                    // Check neighbor bounds
                    if (nx >= 0 && nx < width &&
                        ny >= 0 && ny < height &&
                        nz >= 0 && nz < depth) {
                        
                        int nidx = nz * (width * height) + 
                                   ny * width + nx;
                        sum += input[nidx];
                        count++;
                    }
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

// Launch configuration
dim3 blockSize(8, 8, 8);  // 512 threads per block
dim3 gridSize((width + 7) / 8, 
              (height + 7) / 8,
              (depth + 7) / 8);

process3DVolume<<<gridSize, blockSize>>>(d_input, d_output,
                                          width, height, depth);
```

### 3D Example: Video Processing

```
┌──────────────────────────────────────────────────────────────────────────┐
│              3D INDEXING: VIDEO FRAME PROCESSING                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Problem: Process video as 3D array (time × height × width)              │
│  Video: 30 frames × 1080 height × 1920 width                             │
│                                                                           │
│  3D VIDEO REPRESENTATION:                                                 │
│  ┌────────────────────────────────────────────────────┐                  │
│  │   Time (Z-axis / Frame index)                      │                  │
│  │    ↗                                                │                  │
│  │   /                                                 │                  │
│  │  /  Frame 29                                        │                  │
│  │ /   ┌──────────────────┐                           │                  │
│  │     │  1920 × 1080     │                           │                  │
│  │     │  pixels          │                           │                  │
│  │     ├──────────────────┤                           │                  │
│  │     │  Frame 28        │                           │                  │
│  │     ├──────────────────┤                           │                  │
│  │     │  Frame 27        │                           │                  │
│  │     ├──────────────────┤                           │                  │
│  │     │  ...             │                           │                  │
│  │     ├──────────────────┤                           │                  │
│  │     │  Frame 1         │                           │                  │
│  │     ├──────────────────┤                           │                  │
│  │     │  Frame 0         │                           │                  │
│  │     └──────────────────┘                           │                  │
│  │           │                                         │                  │
│  │           └─→ Width (X-axis)                       │                  │
│  │           ↓                                         │                  │
│  │      Height (Y-axis)                                │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  USE CASES:                                                               │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  1. Temporal filtering (blur across frames)        │                  │
│  │  2. Motion detection (compare adjacent frames)     │                  │
│  │  3. Optical flow computation                        │                  │
│  │  4. Video stabilization                             │                  │
│  │  5. Object tracking across frames                   │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
│  THREAD MAPPING:                                                          │
│  ┌────────────────────────────────────────────────────┐                  │
│  │  Block size: 16×16×2 (512 threads)                 │                  │
│  │  • 16×16 spatial region                            │                  │
│  │  • 2 frames at a time                              │                  │
│  │                                                     │                  │
│  │  Grid size:                                         │                  │
│  │  • X: (1920 + 15) / 16 = 120 blocks                │                  │
│  │  • Y: (1080 + 15) / 16 = 68 blocks                 │                  │
│  │  • Z: (30 + 1) / 2 = 15 blocks                     │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example (Temporal Blur):
──────────────────────────────

__global__ void temporalBlur3D(unsigned char *video, 
                               unsigned char *output,
                               int width, int height, int frames) {
    // Calculate 3D position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;  // time/frame
    
    if (x < width && y < height && t < frames) {
        // Calculate linear index
        int idx = t * (width * height) + y * width + x;
        
        // Temporal blur: average across 3 frames
        float sum = 0.0f;
        int count = 0;
        
        for (int dt = -1; dt <= 1; dt++) {
            int frame = t + dt;
            
            if (frame >= 0 && frame < frames) {
                int tidx = frame * (width * height) + y * width + x;
                sum += video[tidx];
                count++;
            }
        }
        
        output[idx] = (unsigned char)(sum / count);
    }
}

// Launch
dim3 blockSize(16, 16, 2);
dim3 gridSize((width + 15) / 16,
              (height + 15) / 16,
              (frames + 1) / 2);

temporalBlur3D<<<gridSize, blockSize>>>(d_video, d_output,
                                         width, height, frames);
```

---

## Advanced Patterns

### Pattern 1: Checkerboard Access

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CHECKERBOARD ACCESS PATTERN                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Use Case: Red-black Gauss-Seidel, avoiding race conditions              │
│                                                                           │
│  CHECKERBOARD PATTERN (8×8):                                              │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                              │
│  │░░│  │░░│  │░░│  │░░│  │  Row 7                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │  │░░│  │░░│  │░░│  │░░│  Row 6                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │░░│  │░░│  │░░│  │░░│  │  Row 5                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │  │░░│  │░░│  │░░│  │░░│  Row 4                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │░░│  │░░│  │░░│  │░░│  │  Row 3                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │  │░░│  │░░│  │░░│  │░░│  Row 2                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │░░│  │░░│  │░░│  │░░│  │  Row 1                                       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │
│  │  │░░│  │░░│  │░░│  │░░│  Row 0                                       │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                                              │
│   0  1  2  3  4  5  6  7                                                 │
│                                                                           │
│  ░░ = "Red" cells (process first)                                        │
│     = "Black" cells (process second)                                     │
│                                                                           │
│  PHASE 1: Process all red cells (no dependencies)                        │
│  PHASE 2: Synchronize                                                    │
│  PHASE 3: Process all black cells (use updated red values)               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example:
─────────────

__global__ void checkerboardUpdate(float *grid, int width, int height,
                                   int phase) {  // 0=red, 1=black
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Determine if this is a red or black cell
        int cellColor = (x + y) % 2;
        
        // Only process cells matching current phase
        if (cellColor == phase) {
            int idx = y * width + x;
            
            // Update using neighbors (Jacobi/Gauss-Seidel iteration)
            float sum = 0.0f;
            if (x > 0)           sum += grid[idx - 1];
            if (x < width - 1)   sum += grid[idx + 1];
            if (y > 0)           sum += grid[idx - width];
            if (y < height - 1)  sum += grid[idx + width];
            
            grid[idx] = sum * 0.25f;
        }
    }
}

// Launch twice per iteration
checkerboardUpdate<<<grid, block>>>(d_grid, width, height, 0);  // Red
cudaDeviceSynchronize();
checkerboardUpdate<<<grid, block>>>(d_grid, width, height, 1);  // Black
```

### Pattern 2: Tiled Processing with Halo

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  TILED PROCESSING WITH HALO REGIONS                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Use Case: Stencil operations, convolutions, image filters               │
│                                                                           │
│  WITHOUT HALO (Inefficient):                                              │
│  ┌────────────┬────────────┬────────────┐                                │
│  │  Block 0   │  Block 1   │  Block 2   │                                │
│  │  ┌──────┐  │  ┌──────┐  │  ┌──────┐  │                                │
│  │  │ Data │  │  │ Data │  │  │ Data │  │                                │
│  │  └──────┘  │  └──────┘  │  └──────┘  │                                │
│  │     ↓      │     ↓      │     ↓      │                                │
│  │  Multiple  │  Multiple  │  Multiple  │                                │
│  │  global    │  global    │  global    │                                │
│  │  reads     │  reads     │  reads     │                                │
│  └────────────┴────────────┴────────────┘                                │
│                                                                           │
│  WITH HALO (Efficient):                                                   │
│  ┌─────────────────┬─────────────────┬─────────────────┐                │
│  │     Block 0     │     Block 1     │     Block 2     │                │
│  │  ┌───────────┐  │  ┌───────────┐  │  ┌───────────┐  │                │
│  │  │H│  Data │H│  │  │H│  Data │H│  │  │H│  Data │H│  │                │
│  │  │A│       │L│  │  │A│       │L│  │  │A│       │L│  │                │
│  │  │L│       │O│  │  │L│       │O│  │  │L│       │O│  │                │
│  │  │O│       │  │  │  │O│       │  │  │  │O│       │  │  │                │
│  │  └───────────┘  │  └───────────┘  │  └───────────┘  │                │
│  │  Load neighbors │  Load neighbors │  Load neighbors │                │
│  │  once into      │  once into      │  once into      │                │
│  │  shared memory  │  shared memory  │  shared memory  │                │
│  └─────────────────┴─────────────────┴─────────────────┘                │
│                                                                           │
│  DETAILED VIEW OF ONE BLOCK:                                              │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Shared Memory (18×18 for 16×16 block, radius=1) │                    │
│  │                                                    │                    │
│  │  ┌─────────────────────────────────────────┐     │                    │
│  │  │ H H H H H H H H H H H H H H H H H H │     │                    │
│  │  │ A L O ──────────────────────────→   │     │                    │
│  │  │ L       ┌─────────────────────┐   H │     │                    │
│  │  │ O       │                     │   A │     │                    │
│  │  │         │   Core 16×16        │   L │     │                    │
│  │  │         │   threads           │   O │     │                    │
│  │  │         │                     │     │     │                    │
│  │  │         │                     │     │     │                    │
│  │  │         └─────────────────────┘     │     │                    │
│  │  │ H H H H H H H H H H H H H H H H H H │     │                    │
│  │  │ A L O ──────────────────────────→   │     │                    │
│  │  └─────────────────────────────────────────┘     │                    │
│  └──────────────────────────────────────────────────┘                    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Code Example (3x3 Convolution):
────────────────────────────────

#define TILE_SIZE 16
#define RADIUS 1
#define BLOCK_SIZE (TILE_SIZE + 2*RADIUS)

__global__ void convolve2DHalo(float *input, float *output,
                               int width, int height) {
    __shared__ float shared[BLOCK_SIZE][BLOCK_SIZE];
    
    // Global position
    int col = blockIdx.x * TILE_SIZE + threadIdx.x - RADIUS;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y - RADIUS;
    
    // Shared memory position
    int s_col = threadIdx.x;
    int s_row = threadIdx.y;
    
    // Load data with halo into shared memory
    if (col >= 0 && col < width && row >= 0 && row < height) {
        shared[s_row][s_col] = input[row * width + col];
    } else {
        shared[s_row][s_col] = 0.0f;  // Padding
    }
    
    __syncthreads();
    
    // Only compute for core region (not halo)
    if (threadIdx.x >= RADIUS && threadIdx.x < BLOCK_SIZE - RADIUS &&
        threadIdx.y >= RADIUS && threadIdx.y < BLOCK_SIZE - RADIUS) {
        
        col = blockIdx.x * TILE_SIZE + threadIdx.x - RADIUS;
        row = blockIdx.y * TILE_SIZE + threadIdx.y - RADIUS;
        
        if (col < width && row < height) {
            float sum = 0.0f;
            
            // 3×3 convolution from shared memory
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    sum += shared[s_row + dy][s_col + dx];
                }
            }
            
            output[row * width + col] = sum / 9.0f;  // Average
        }
    }
}

// Launch
dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

convolve2DHalo<<<gridSize, blockSize>>>(d_input, d_output, width, height);
```

---

## Common Pitfalls

### Pitfall 1: Off-by-One Errors

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      OFF-BY-ONE ERRORS                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  WRONG: Missing boundary check                                           │
│  ────────────────────────────                                             │
│  __global__ void kernel(float *data, int N) {                            │
│      int idx = blockIdx.x * blockDim.x + threadIdx.x;                    │
│      data[idx] = idx * 2.0f;  // ← BUG! May exceed N                     │
│  }                                                                        │
│                                                                           │
│  Problem: If N=100, blockSize=32, you launch 4 blocks (128 threads)      │
│  Threads 100-127 will access out-of-bounds memory!                       │
│                                                                           │
│  CORRECT: Always check bounds                                             │
│  ─────────────────────────────                                            │
│  __global__ void kernel(float *data, int N) {                            │
│      int idx = blockIdx.x * blockDim.x + threadIdx.x;                    │
│      if (idx < N) {  // ← Boundary check                                 │
│          data[idx] = idx * 2.0f;                                          │
│      }                                                                    │
│  }                                                                        │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pitfall 2: Row-Major vs Column-Major

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  ROW-MAJOR vs COLUMN-MAJOR CONFUSION                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  C/CUDA uses ROW-MAJOR order:                                            │
│  ┌──────────────────────────────────────────┐                            │
│  │  Matrix[row][col]:                        │                            │
│  │  ┌───┬───┬───┬───┐                        │                            │
│  │  │ 0 │ 1 │ 2 │ 3 │  Row 0                 │                            │
│  │  ├───┼───┼───┼───┤                        │                            │
│  │  │ 4 │ 5 │ 6 │ 7 │  Row 1                 │                            │
│  │  └───┴───┴───┴───┘                        │                            │
│  │                                            │                            │
│  │  Memory layout: [0,1,2,3,4,5,6,7]        │                            │
│  │                  └─Row 0─┘└─Row 1─┘      │                            │
│  │                                            │                            │
│  │  Access: index = row * width + col        │                            │
│  └──────────────────────────────────────────┘                            │
│                                                                           │
│  FORTRAN/MATLAB use COLUMN-MAJOR order:                                  │
│  ┌──────────────────────────────────────────┐                            │
│  │  Matrix[row][col]:                        │                            │
│  │  ┌───┬───┬───┬───┐                        │                            │
│  │  │ 0 │ 2 │ 4 │ 6 │  Row 0                 │                            │
│  │  ├───┼───┼───┼───┤                        │                            │
│  │  │ 1 │ 3 │ 5 │ 7 │  Row 1                 │                            │
│  │  └───┴───┴───┴───┘                        │                            │
│  │                                            │                            │
│  │  Memory layout: [0,1,2,3,4,5,6,7]        │                            │
│  │                  └Col0┘└Col1┘└Col2┘└Col3┘│                            │
│  │                                            │                            │
│  │  Access: index = col * height + row       │                            │
│  └──────────────────────────────────────────┘                            │
│                                                                           │
│  CORRECT for C/CUDA:                                                      │
│  int idx = row * width + col;  // Row-major                              │
│                                                                           │
│  WRONG (will access wrong elements):                                      │
│  int idx = col * height + row;  // Column-major (FORTRAN style)          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pitfall 3: X/Y Axis Confusion

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       X/Y AXIS CONFUSION                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  COMMON MISTAKE: Swapping X and Y                                        │
│                                                                           │
│  WRONG:                                                                   │
│  ──────                                                                   │
│  int col = blockIdx.y * blockDim.y + threadIdx.y;  // ← WRONG!           │
│  int row = blockIdx.x * blockDim.x + threadIdx.x;  // ← WRONG!           │
│                                                                           │
│  This maps:                                                               │
│  • X-axis (usually horizontal/width) → rows (vertical)                   │
│  • Y-axis (usually vertical/height) → cols (horizontal)                  │
│                                                                           │
│  CORRECT:                                                                 │
│  ────────                                                                 │
│  int col = blockIdx.x * blockDim.x + threadIdx.x;  // X → columns        │
│  int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y → rows           │
│                                                                           │
│  VISUAL REMINDER:                                                         │
│  ┌────────────────────────────────────────┐                              │
│  │         X-axis (width, columns)        │                              │
│  │       ─────────────────────→            │                              │
│  │  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐         │                              │
│  │  Y  │  │  │  │  │  │  │  │  │         │                              │
│  │  -  ├──┼──┼──┼──┼──┼──┼──┼──┤         │                              │
│  │  a  │  │  │  │  │  │  │  │  │         │                              │
│  │  x  ├──┼──┼──┼──┼──┼──┼──┼──┤         │                              │
│  │  i  │  │  │  │  │  │  │  │  │         │                              │
│  │  s  ├──┼──┼──┼──┼──┼──┼──┼──┤         │                              │
│  │  │  │  │  │  │  │  │  │  │  │         │                              │
│  │  ↓  └──┴──┴──┴──┴──┴──┴──┴──┘         │                              │
│  │  (height, rows)                        │                              │
│  └────────────────────────────────────────┘                              │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Coalesced Memory Access

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   MEMORY COALESCING IN 2D/3D                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  GOOD: Consecutive threads access consecutive memory                     │
│  ────────────────────────────────────────────────────                     │
│  Row-wise access (X-direction):                                          │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                              │
│  │T0│T1│T2│T3│T4│T5│T6│T7│  ← Warp of threads                           │
│  └┬─┴┬─┴┬─┴┬─┴┬─┴┬─┴┬─┴┬─┘                                              │
│   ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓                                                │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                              │
│  │0 │1 │2 │3 │4 │5 │6 │7 │  Memory (consecutive)                        │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                                              │
│  Result: Single memory transaction (coalesced) ✓                         │
│                                                                           │
│  BAD: Strided or column-wise access                                      │
│  ───────────────────────────────────                                      │
│  Column-wise access (Y-direction):                                       │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                              │
│  │T0│T1│T2│T3│T4│T5│T6│T7│  ← Warp of threads                           │
│  └┬─┴┬─┴┬─┴┬─┴┬─┴┬─┴┬─┴┬─┘                                              │
│   ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓                                                │
│  [0]-│--│--│--│--│--│--│--│                                              │
│  [W]-┘--│--│--│--│--│--│--│                                              │
│  [2W]---┘--│--│--│--│--│--│  (W = width, large stride)                  │
│  [3W]------┘--│--│--│--│--│                                              │
│  [4W]---------┘--│--│--│--│                                              │
│  [5W]------------┘--│--│--│                                              │
│  [6W]---------------┘--│--│                                              │
│  [7W]------------------┘--│                                              │
│  Result: Multiple memory transactions (uncoalesced) ✗                    │
│                                                                           │
│  FOR BEST PERFORMANCE:                                                    │
│  • Process data row-wise (increment X fastest)                           │
│  • Use shared memory for column-wise access                              │
│  • Transpose data if column-wise processing is unavoidable               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Index Calculation Formulas

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    INDEX CALCULATION REFERENCE                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1D INDEXING:                                                             │
│  ────────────                                                             │
│  globalIdx = blockIdx.x * blockDim.x + threadIdx.x                       │
│                                                                           │
│  2D INDEXING:                                                             │
│  ────────────                                                             │
│  col = blockIdx.x * blockDim.x + threadIdx.x                             │
│  row = blockIdx.y * blockDim.y + threadIdx.y                             │
│  linearIdx = row * width + col                                           │
│                                                                           │
│  3D INDEXING:                                                             │
│  ────────────                                                             │
│  x = blockIdx.x * blockDim.x + threadIdx.x                               │
│  y = blockIdx.y * blockDim.y + threadIdx.y                               │
│  z = blockIdx.z * blockDim.z + threadIdx.z                               │
│  linearIdx = z * (width * height) + y * width + x                        │
│                                                                           │
│  GRID-STRIDE LOOP (1D):                                                   │
│  ──────────────────────                                                   │
│  int idx = blockIdx.x * blockDim.x + threadIdx.x;                        │
│  int stride = blockDim.x * gridDim.x;                                    │
│  for (int i = idx; i < N; i += stride) {                                 │
│      // process data[i]                                                   │
│  }                                                                        │
│                                                                           │
│  COMMON BLOCK/GRID SIZES:                                                 │
│  ─────────────────────────                                                │
│  1D: 256, 512, 1024 threads                                              │
│  2D: 16×16 (256), 32×32 (1024)                                           │
│  3D: 8×8×8 (512), 16×16×4 (1024)                                         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This guide covered:

✓ **1D Thread Indexing** - Vector operations, strided access  
✓ **2D Thread Indexing** - Image processing, matrix operations  
✓ **3D Thread Indexing** - Volume processing, video processing  
✓ **Advanced Patterns** - Checkerboard, tiled with halo  
✓ **Common Pitfalls** - Boundary checks, row/column-major, axis confusion  
✓ **Performance Tips** - Memory coalescing, optimal access patterns  

### Key Takeaways

1. **Always check boundaries** - `if (idx < N)` prevents out-of-bounds access
2. **Match dimensionality to data** - Use 2D for images, 3D for volumes
3. **Remember row-major order** - `index = row * width + col` in C/CUDA
4. **Coalesce memory access** - Process along X-axis for best performance
5. **Use appropriate block sizes** - 256-1024 threads, consider occupancy

### Next Steps

- Practice with the code examples in [04_thread_organization.cu](04_thread_organization.cu)
- Implement custom kernels for your specific data layout
- Profile memory access patterns with Nsight Compute
- Study shared memory patterns in [06_shared_memory.cu](06_shared_memory.cu)

---

*For executable code examples, see [04_thread_organization.cu](04_thread_organization.cu)*

*For performance profiling, see [09_profiling_debugging.md](09_profiling_debugging.md)*

