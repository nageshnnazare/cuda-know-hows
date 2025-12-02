# CUDA Work Allocation Guide
## When and Where to Use Blocks, Grids, and Threads

### The Complete Guide to Efficient Work Distribution

---

## Table of Contents

1. [Hardware Hierarchy](#hardware-hierarchy)
2. [Mapping Work to Hardware](#mapping-work)
3. [Block Size Selection](#block-size)
4. [Grid Size Selection](#grid-size)
5. [Occupancy Optimization](#occupancy)
6. [Work Distribution Patterns](#patterns)
7. [Real-World Examples](#examples)
8. [Decision Framework](#decision-framework)

---

## Hardware Hierarchy {#hardware-hierarchy}

### Understanding the GPU Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    GPU HARDWARE HIERARCHY                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  GPU Device (e.g., RTX 3090)                                         ║
║  ┌────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Graphics Processing Cluster (GPC) × 7                           │ ║
║  │  ┌────────────────────────────────────────────────────────────┐ │ ║
║  │  │                                                              │ │ ║
║  │  │  Texture Processing Cluster (TPC) × 6                        │ │ ║
║  │  │  ┌──────────────────────────────────────────────────────┐  │ │ ║
║  │  │  │                                                        │  │ │ ║
║  │  │  │  Streaming Multiprocessor (SM) × 2                     │  │ │ ║
║  │  │  │  ┌──────────────────────────────────────────────────┐ │  │ │ ║
║  │  │  │  │                                                    │ │  │ │ ║
║  │  │  │  │  Warp Schedulers × 4                               │ │  │ │ ║
║  │  │  │  │  ┌───────────────────────────────────────────┐   │ │  │ │ ║
║  │  │  │  │  │                                             │   │ │  │ │ ║
║  │  │  │  │  │  Warp (32 threads)                          │   │ │  │ │ ║
║  │  │  │  │  │  ┌─────────────────────────────────────┐  │   │ │  │ │ ║
║  │  │  │  │  │  │  CUDA Core × 32                     │  │   │ │  │ │ ║
║  │  │  │  │  │  │  (executes 1 thread each)           │  │   │ │  │ │ ║
║  │  │  │  │  │  └─────────────────────────────────────┘  │   │ │  │ │ ║
║  │  │  │  │  │                                             │   │ │  │ │ ║
║  │  │  │  │  │  Each warp scheduler can manage            │   │ │  │ │ ║
║  │  │  │  │  │  up to 16 warps simultaneously             │   │ │  │ │ ║
║  │  │  │  │  └───────────────────────────────────────────┘   │ │  │ │ ║
║  │  │  │  │                                                    │ │  │ │ ║
║  │  │  │  │  Per SM Resources:                                │ │  │ │ ║
║  │  │  │  │  • 128 CUDA cores (FP32)                          │ │  │ │ ║
║  │  │  │  │  • 64 FP64 cores                                  │ │  │ │ ║
║  │  │  │  │  • 65,536 registers                               │ │  │ │ ║
║  │  │  │  │  • 128 KB shared memory/L1 cache                  │ │  │ │ ║
║  │  │  │  │  • Max 1024 threads resident                      │ │  │ │ ║
║  │  │  │  │  • Max 16 blocks resident                         │ │  │ │ ║
║  │  │  │  └──────────────────────────────────────────────────┘ │  │ │ ║
║  │  │  │                                                        │  │ │ ║
║  │  │  └────────────────────────────────────────────────────────┘  │ │ ║
║  │  │                                                              │ │ ║
║  │  └──────────────────────────────────────────────────────────────┘ │ ║
║  │                                                                  │ ║
║  └────────────────────────────────────────────────────────────────┘ ║
║                                                                      ║
║  Total for RTX 3090: 82 SMs × 128 cores = 10,496 CUDA cores         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Key Concepts

**Thread**: Single execution unit (runs on 1 CUDA core)  
**Warp**: 32 threads that execute together (SIMT)  
**Block**: Group of threads (up to 1024) sharing resources  
**Grid**: Collection of blocks that execute a kernel  
**SM**: Physical processor that executes blocks  

---

## Mapping Work to Hardware {#mapping-work}

### The Complete Mapping

```
╔════════════════════════════════════════════════════════════════╗
║            SOFTWARE → HARDWARE MAPPING                         ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  YOUR CODE:              HARDWARE:                             ║
║  ─────────              ─────────                             ║
║                                                                ║
║  Grid                                                          ║
║  ├─ Block 0    ────→    SM 0                                  ║
║  ├─ Block 1    ────→    SM 1                                  ║
║  ├─ Block 2    ────→    SM 0 (when Block 0 finishes)          ║
║  ├─ Block 3    ────→    SM 2                                  ║
║  ├─ Block 4    ────→    SM 1 (when Block 1 finishes)          ║
║  └─ ...                                                        ║
║                                                                ║
║  Block (256 threads)                                           ║
║  ├─ Warp 0 (threads 0-31)    ────→  Warp Scheduler 0          ║
║  ├─ Warp 1 (threads 32-63)   ────→  Warp Scheduler 1          ║
║  ├─ Warp 2 (threads 64-95)   ────→  Warp Scheduler 2          ║
║  └─ ...                                                        ║
║                                                                ║
║  Thread ────→ CUDA Core (when scheduled)                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Execution Timeline

```
Time →
════════════════════════════════════════════════════════════════

SM 0:  [Block 0────]  [Block 2────]  [Block 5────] ...
SM 1:  [Block 1────]  [Block 3────]  [Block 6────] ...
SM 2:  [Block 4────]  [Block 7────]  [Block 9────] ...
...

Within Block 0 on SM 0:
─────────────────────
Cycle 0-31:   Warp 0 executes (threads 0-31 on 32 cores)
Cycle 32-63:  Warp 1 executes (threads 32-63)
Cycle 64-95:  Warp 2 executes (threads 64-95)
...
(Warps can interleave based on dependencies/memory latency)
```

### Resource Allocation

```
╔════════════════════════════════════════════════════════════════╗
║                  PER-SM RESOURCE LIMITS                        ║
║                  (Example: Ampere Architecture)                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Resource                  Limit         Shared By             ║
║  ────────────────────────  ──────────    ──────────            ║
║  Max Threads               2048          All blocks on SM      ║
║  Max Blocks                16            All warps on SM       ║
║  Max Warps                 64            All blocks on SM      ║
║  Registers                 65536         All threads on SM     ║
║  Shared Memory             100 KB        All blocks on SM      ║
║  Shared Mem per Block      48 KB (max)   Single block          ║
║                                                                ║
║  Example: If your block uses:                                  ║
║  • 256 threads                                                 ║
║  • 32 registers per thread                                     ║
║  • 16 KB shared memory                                         ║
║                                                                ║
║  Then per SM:                                                  ║
║  • Thread limit: 2048 / 256 = 8 blocks                         ║
║  • Register limit: 65536 / (256 × 32) = 8 blocks               ║
║  • Shared mem limit: 100KB / 16KB = 6 blocks ← BOTTLENECK      ║
║  → Only 6 blocks can run simultaneously per SM                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## Block Size Selection {#block-size}

### Block Size Constraints

```
╔════════════════════════════════════════════════════════════════╗
║                    BLOCK SIZE RULES                            ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Hard Limits:                                                  ║
║  • Min: 1 thread (not recommended!)                            ║
║  • Max: 1024 threads per block                                 ║
║  • Must be multiple of warp size (32) for efficiency           ║
║                                                                ║
║  Recommended Range:                                            ║
║  • 128-512 threads per block                                   ║
║  • Sweet spot: 256 threads (8 warps)                           ║
║                                                                ║
║  Why multiples of 32?                                          ║
║  Block with 100 threads:                                       ║
║    Warp 0: 32 threads ████████████████                         ║
║    Warp 1: 32 threads ████████████████                         ║
║    Warp 2: 32 threads ████████████████                         ║
║    Warp 3:  4 threads ████░░░░░░░░░░░░ ← 28 cores wasted!     ║
║                                                                ║
║  Block with 128 threads (better):                              ║
║    Warp 0: 32 threads ████████████████ ✓                      ║
║    Warp 1: 32 threads ████████████████ ✓                      ║
║    Warp 2: 32 threads ████████████████ ✓                      ║
║    Warp 3: 32 threads ████████████████ ✓                      ║
║    All cores utilized!                                         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Block Size Trade-offs

```
Small Blocks (32-64 threads):
════════════════════════════
✓ More blocks fit on SM
✓ Better load balancing
✓ Hide latency better
✗ Less shared memory per thread
✗ More kernel launch overhead

Medium Blocks (128-256 threads):  ← RECOMMENDED
═══════════════════════════════
✓ Good occupancy
✓ Efficient shared memory use
✓ Good latency hiding
✓ Balanced resource usage

Large Blocks (512-1024 threads):
════════════════════════════════
✓ Maximum shared memory per block
✓ Good for reduction operations
✗ Fewer blocks per SM
✗ Reduced occupancy if resource-heavy
✗ Less flexible scheduling
```

### Decision Matrix for Block Size

```
╔═══════════════════════════════════════════════════════════════════╗
║ Workload Type          │ Recommended Block Size │ Why?           ║
╠════════════════════════╪════════════════════════╪════════════════╣
║ Simple compute         │ 256                    │ Balanced       ║
║ (element-wise ops)     │                        │                ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ Heavy shared memory    │ 128-256                │ Limit shared   ║
║ (matrix multiply)      │                        │ mem per block  ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ Many registers         │ 128-192                │ Avoid register ║
║ (complex math)         │                        │ spilling       ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ Reduction/scan         │ 256-512                │ More threads   ║
║                        │                        │ = faster tree  ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ Image processing       │ 16×16 (256)            │ 2D spatial     ║
║                        │                        │ locality       ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ 3D volume              │ 8×8×4 (256)            │ 3D spatial     ║
║                        │                        │ locality       ║
║ ───────────────────────┼────────────────────────┼────────────────║
║ Memory-bound           │ 256-512                │ Hide latency   ║
║                        │                        │ with warps     ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Grid Size Selection {#grid-size}

### Grid Sizing Strategies

#### **Strategy 1: Cover All Elements**

```cpp
// Problem: Process N elements
int N = 10000000;  // 10M elements

// Block size: 256 threads
int blockSize = 256;

// Grid size: Enough blocks to cover all elements
int gridSize = (N + blockSize - 1) / blockSize;
// = (10000000 + 255) / 256
// = 39063 blocks

// Launch
kernel<<<gridSize, blockSize>>>(data, N);

// Visualization:
// ┌────────────────────────────────────────────┐
// │ Thread Coverage:                           │
// │ Block 0:    Elements 0-255                 │
// │ Block 1:    Elements 256-511               │
// │ Block 2:    Elements 512-767               │
// │ ...                                        │
// │ Block 39062: Elements 9999872-9999999      │
// └────────────────────────────────────────────┘
```

#### **Strategy 2: Saturate GPU**

```cpp
// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

int numSMs = prop.multiProcessorCount;  // e.g., 82 for RTX 3090
int maxBlocksPerSM = 16;  // Architecture dependent

// Saturate GPU with enough blocks
int gridSize = numSMs * maxBlocksPerSM;  // 82 × 16 = 1312 blocks

// Each block processes multiple elements
int elementsPerBlock = (N + gridSize - 1) / gridSize;

// Visualization:
// GPU with 4 SMs, 2 blocks per SM:
// ┌─────────┬─────────┬─────────┬─────────┐
// │ SM 0    │ SM 1    │ SM 2    │ SM 3    │
// ├─────────┼─────────┼─────────┼─────────┤
// │ Block 0 │ Block 2 │ Block 4 │ Block 6 │
// │ Block 1 │ Block 3 │ Block 5 │ Block 7 │
// └─────────┴─────────┴─────────┴─────────┘
// All SMs fully utilized!
```

#### **Strategy 3: Grid-Stride Loop** (Most Flexible!)

```cpp
__global__ void gridStrideKernel(float *data, int N) {
    // Global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stride = total number of threads in grid
    int stride = blockDim.x * gridDim.x;
    
    // Loop with grid stride
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}

// Can launch with ANY grid size!
kernel<<<100, 256>>>(data, 10000000);    // Works
kernel<<<1000, 256>>>(data, 10000000);   // Works
kernel<<<numSMs * 8, 256>>>(data, 10000000);  // Optimal

// Visualization (4 blocks, 8 elements):
// Elements: [0] [1] [2] [3] [4] [5] [6] [7]
//            ↑       ↑       ↑       ↑
//         Blk 0   Blk 1   Blk 2   Blk 3  (first iteration)
//         Then Block 0 processes element 4, etc.
```

### Grid Size Comparison

```
╔═══════════════════════════════════════════════════════════════════╗
║ Strategy        │ Grid Size      │ Pros              │ Cons       ║
╠═════════════════╪════════════════╪═══════════════════╪════════════╣
║ Exact coverage  │ N/blockSize    │ Simple            │ May have   ║
║                 │                │ No loops needed   │ too many   ║
║                 │                │                   │ blocks     ║
║ ────────────────┼────────────────┼───────────────────┼────────────║
║ Saturate GPU    │ numSMs × k     │ Good occupancy    │ Needs loop ║
║                 │ (k=8-16)       │ Less overhead     │ if N large ║
║ ────────────────┼────────────────┼───────────────────┼────────────║
║ Grid-stride     │ Flexible       │ Works for any N   │ Loop       ║
║                 │ (numSMs × 8)   │ Good balance      │ overhead   ║
║                 │                │ Reusable code     │            ║
╚═══════════════════════════════════════════════════════════════════╝

Recommendation: Use grid-stride for maximum flexibility!
```

---

## Occupancy Optimization {#occupancy}

### What is Occupancy?

```
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)

Example:
─────── SM can handle: 64 warps maximum
Currently running: 48 warps
Occupancy = 48 / 64 = 75%
```

### Occupancy Impact

```
╔════════════════════════════════════════════════════════════════╗
║                    OCCUPANCY EFFECTS                           ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Low Occupancy (25%):        High Occupancy (75%):             ║
║  ────────────────────        ─────────────────────            ║
║  SM Resources:               SM Resources:                     ║
║  ┌─────────────────┐        ┌─────────────────┐              ║
║  │ ████░░░░░░░░░░░░│        │ ████████████░░░░│              ║
║  │ 25% utilized    │        │ 75% utilized    │              ║
║  └─────────────────┘        └─────────────────┘              ║
║                                                                ║
║  Latency Hiding:             Latency Hiding:                   ║
║  Warp 0: [Memory wait...]    Warp 0:  [Memory wait]            ║
║  Warp 1: [Idle]              Warp 1:  [Compute]                ║
║  Warp 2: [Idle]              Warp 2:  [Compute]                ║
║  ...                         Warp 3:  [Compute]                ║
║  ← Poor latency hiding       ← Good latency hiding!            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Computing Occupancy

```cpp
// Method 1: CUDA Occupancy API
int blockSize = 256;
int numBlocks;
int minGridSize;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    myKernel,
    0,  // Dynamic shared memory per block
    0   // Block size limit
);

printf("Recommended block size: %d\n", blockSize);
printf("Minimum grid size for max occupancy: %d\n", minGridSize);

// Method 2: Manual calculation
int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;  // e.g., 2048
int threadsPerBlock = 256;
int blocksPerSM = maxThreadsPerSM / threadsPerBlock;  // 2048 / 256 = 8

// But also limited by registers, shared memory!
```

### Occupancy Limiting Factors

```
Example Kernel with 256 threads/block:
═══════════════════════════════════════

Factor 1: Threads
───────────────
Max threads per SM: 2048
Threads per block: 256
→ Max blocks: 2048 / 256 = 8 blocks

Factor 2: Registers (32 registers/thread)
────────────────────────────────────────
Total registers: 65536
Per block: 256 threads × 32 regs = 8192 registers
→ Max blocks: 65536 / 8192 = 8 blocks ✓

Factor 3: Shared Memory (20 KB/block)
──────────────────────────────────────
Total shared memory: 100 KB
Per block: 20 KB
→ Max blocks: 100KB / 20KB = 5 blocks ← BOTTLENECK!

RESULT: Only 5 blocks per SM
        Occupancy = (5 × 256) / 2048 = 62.5%
```

### Optimizing for Higher Occupancy

```cpp
// Before: Low occupancy due to shared memory
__global__ void lowOccupancy(float *data, int n) {
    __shared__ float buffer[8192];  // 32 KB!
    // Only 3 blocks per SM (100KB / 32KB)
    // Occupancy = 37.5%
}

// After: Reduced shared memory
__global__ void highOccupancy(float *data, int n) {
    __shared__ float buffer[2048];  // 8 KB
    // Now 12 blocks per SM (100KB / 8KB)
    // Occupancy = 75%
}

// Alternative: Process in chunks
__global__ void chunked(float *data, int n) {
    __shared__ float buffer[2048];  // 8 KB
    
    // Process multiple chunks per block
    for (int chunk = 0; chunk < 4; chunk++) {
        // Load chunk into shared memory
        // Process
        // Repeat
    }
}
```

### Occupancy vs Performance

```
⚠️ IMPORTANT: High occupancy ≠ High performance!

Scenario 1: Compute-bound kernel
────────────────────────────────
50% occupancy is often enough!
More occupancy doesn't help if ALUs are already saturated.

Scenario 2: Memory-bound kernel
────────────────────────────────
Higher occupancy helps hide memory latency.
Aim for 75%+ occupancy.

Rule of Thumb:
─────────────
• Occupancy > 50%: Usually sufficient
• Occupancy 60-75%: Sweet spot
• Occupancy > 75%: Diminishing returns
• Occupancy 100%: Not always better!

Profile First! Don't optimize blindly.
```

---

## Work Distribution Patterns {#patterns}

### Pattern 1: Element-wise Operations (Simplest)

```cpp
// Each thread processes exactly ONE element
__global__ void elementWise(float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Work Distribution:
// N = 1000 elements
// blockSize = 256
// gridSize = (1000 + 255) / 256 = 4 blocks
//
// Block 0: Elements 0-255     (256 elements)
// Block 1: Elements 256-511   (256 elements)
// Block 2: Elements 512-767   (256 elements)
// Block 3: Elements 768-999   (232 elements)
```

### Pattern 2: 2D Image Processing

```cpp
// Each thread processes ONE pixel
__global__ void imageProcess(unsigned char *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = image[idx] / 2;  // Darken
    }
}

// Work Distribution for 1920×1080 image:
// blockSize: 16×16 = 256 threads
// gridSize: (1920/16) × (1080/16) = 120 × 68 = 8,160 blocks
//
// Visual:
// ┌────────────────────────────────────┐
// │ [Block]  [Block]  [Block] ... ×120 │ ← Row 0 of blocks
// │ [Block]  [Block]  [Block] ... ×120 │ ← Row 1
// │ [Block]  [Block]  [Block] ... ×120 │ ← Row 2
// │   ...                              │
// │ ×68 rows of blocks                 │
// └────────────────────────────────────┘
//
// Each block processes 16×16 = 256 pixels
```

### Pattern 3: Grid-Stride (Flexible)

```cpp
__global__ void gridStride(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes MULTIPLE elements
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}

// Example: N=1000, 4 blocks of 64 threads
// Total threads = 4 × 64 = 256
// Stride = 256
//
// Thread 0:   Elements 0, 256, 512, 768
// Thread 1:   Elements 1, 257, 513, 769
// Thread 2:   Elements 2, 258, 514, 770
// ...
// Thread 255: Elements 255, 511, 767, 999
```

### Pattern 4: Tiled with Shared Memory

```cpp
__global__ void tiled(float *in, float *out, int N) {
    __shared__ float tile[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load tile
    if (idx < N) {
        tile[tid] = in[idx];
    }
    __syncthreads();
    
    // Process from shared memory
    if (idx < N) {
        out[idx] = tile[tid] + 
                   (tid > 0 ? tile[tid-1] : 0.0f) +
                   (tid < 255 ? tile[tid+1] : 0.0f);
    }
}

// Work Distribution:
// Each block:
//   1. Loads 256 elements into shared memory (coalesced)
//   2. Processes using fast shared memory
//   3. Writes 256 results (coalesced)
```

### Pattern 5: Reduction (Tree-based)

```cpp
__global__ void reduction(float *input, float *output, int N) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    shared[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Work Distribution:
// Each block reduces 256 elements → 1 value
// If N = 10,000:
//   gridSize = 40 blocks
//   Each block: 256 elements → 1 sum
//   Final: 40 partial sums (reduce again on CPU or GPU)
```

---

## Real-World Examples {#examples}

### Example 1: Vector Addition

```cpp
// Problem: Add two vectors of 10M elements
int N = 10000000;

// Configuration:
dim3 blockSize(256);
dim3 gridSize((N + 255) / 256);  // 39,063 blocks

vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

// GPU Execution (82 SMs):
// ────────────────────────
// Iteration 1: Blocks 0-1311 run simultaneously (82 × 16)
// Iteration 2: Blocks 1312-2623 run
// ...
// Iteration 30: Blocks 39040-39062 run (last 23 blocks)
//
// Total time: ~0.5 ms (memory-bound)
```

### Example 2: Matrix Multiplication (1024×1024)

```cpp
// Problem: C = A × B, all 1024×1024
int M = 1024, K = 1024, N = 1024;

// Configuration:
dim3 blockSize(32, 32);           // 1024 threads per block
dim3 gridSize(N/32, M/32);        // 32 × 32 = 1024 blocks

matmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

// Work Distribution:
// ──────────────────
// Each block: Computes 32×32 output tile
// Requires: Loading tiles from A and B
// Shared memory: 2 × 32×32 × 4 bytes = 8 KB per block
//
// SM Occupancy:
// Max blocks per SM: 16 (limited by max blocks, not resources)
// Threads per SM: 16 × 1024 = 16,384 threads
// But max is 2048, so actually: 2 blocks per SM
// Occupancy = 2048 / 2048 = 100%
```

### Example 3: Image Convolution (1920×1080)

```cpp
// Problem: Apply 5×5 Gaussian blur
int width = 1920, height = 1080;

// Configuration:
dim3 blockSize(16, 16);                     // 256 threads
dim3 gridSize((width+15)/16, (height+15)/16);  // 120 × 68 = 8,160 blocks

convolution<<<gridSize, blockSize>>>(image, width, height);

// Work Distribution:
// ──────────────────
// Each block: 16×16 pixels
// With halo: Actually loads (16+4)×(16+4) = 400 pixels
// Shared memory: 400 bytes per block
//
// SM Assignment (82 SMs):
// Iteration 1: 82 SMs × 16 blocks/SM = 1,312 blocks run
// Iteration 2: Next 1,312 blocks run
// ...
// Iteration 7: Last blocks finish
```

### Example 4: Large Reduction (100M elements)

```cpp
// Problem: Sum 100M elements
int N = 100000000;

// Configuration (two-phase):
int blockSize = 512;                    // Large for reduction
int gridSize = min((N + 511) / 512, 1024);  // Cap at 1024 blocks

reduce<<<gridSize, blockSize>>>(d_input, d_partial, N);
// → Produces 1024 partial sums

reduce<<<1, 512>>>(d_partial, d_final, 1024);
// → Final sum

// Why cap grid size?
// ──────────────────
// 1. Each block produces 1 output
// 2. Too many blocks → too many partials → expensive final reduction
// 3. Grid-stride inside block handles large N
// 4. 1024 blocks fully saturate 82 SMs
```

---

## Decision Framework {#decision-framework}

### Step-by-Step Configuration Guide

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Choose Block Dimensions                                │
└─────────────────────────────────────────────────────────────────┘

Problem Type?
├─ 1D data (vectors) → blockDim.x = 256
├─ 2D data (images)  → blockDim.x = 16, blockDim.y = 16
└─ 3D data (volumes) → blockDim.x = 8, blockDim.y = 8, blockDim.z = 4

Total threads = blockDim.x × blockDim.y × blockDim.z
Ensure: 128 ≤ total ≤ 512 (usually)
Must be: multiple of 32

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Choose Grid Dimensions                                 │
└─────────────────────────────────────────────────────────────────┘

Strategy?
├─ Small N (<1M)  → gridSize = (N + blockSize - 1) / blockSize
├─ Large N (>1M)  → gridSize = numSMs × (8-16)
└─ Variable N     → Use grid-stride loop

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Check Resource Usage                                   │
└─────────────────────────────────────────────────────────────────┘

Compile with:
nvcc --ptxas-options=-v mykernel.cu

Output shows:
• Registers per thread
• Shared memory per block
• Spilled registers (bad!)

If occupancy < 50%:
├─ Reduce shared memory usage
├─ Reduce register usage (simplify kernel)
└─ Decrease block size

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Profile and Iterate                                    │
└─────────────────────────────────────────────────────────────────┘

ncu --metrics sm_efficiency,achieved_occupancy ./myprogram

If performance poor:
├─ Check occupancy (aim for 60%+)
├─ Check memory throughput (compare to peak)
├─ Check warp divergence
└─ Try different block sizes
```

### Decision Tree

```
Choose Block Size:
==================

Is problem 2D/3D?
├─ YES → Use 2D/3D blocks
│        • 2D: 16×16 or 32×32
│        • 3D: 8×8×4 or 4×4×16
│
└─ NO → Use 1D blocks
        │
        Do you use shared memory?
        ├─ YES → How much per thread?
        │        ├─ < 128 bytes → blockSize = 256-512
        │        └─ > 128 bytes → blockSize = 128-256
        │
        └─ NO → Do you use many registers?
                 ├─ YES → blockSize = 128-256
                 └─ NO → blockSize = 256-512

Choose Grid Size:
=================

How large is N?
├─ Small (< 100K) → gridSize = (N + blockSize - 1) / blockSize
│
├─ Medium (100K - 10M) → gridSize = numSMs × 8
│                        Use grid-stride loop
│
└─ Large (> 10M) → gridSize = numSMs × 8-16
                   Use grid-stride loop
```

---

## Practical Examples with Analysis

### Example A: Simple Vector Scaling

```cpp
// Scale 10M elements by 2.0
__global__ void scale(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

// Configuration Analysis:
// ─────────────────────
int N = 10000000;
int blockSize = 256;        // Why 256?
                            // • Multiple of 32 ✓
                            // • No shared memory needed
                            // • Simple kernel → don't need many warps
                            // • 256 is standard default ✓

int gridSize = (N + 255) / 256;  // 39,063 blocks

// Will this saturate GPU?
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int numSMs = prop.multiProcessorCount;  // 82

// Blocks per SM: 39063 / 82 ≈ 476 blocks per SM
// Each SM runs ~16 blocks at a time → Great!
// GPU will be well-saturated ✓
```

### Example B: Matrix Multiplication

```cpp
// Multiply 2048×2048 matrices
__global__ void matmul(float *A, float *B, float *C, int N) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];
    // ... (tiled implementation)
}

// Configuration Analysis:
// ─────────────────────
int N = 2048;
dim3 blockSize(32, 32);     // 1024 threads - Why?
                            // • 2D problem → 2D blocks ✓
                            // • 32×32 = 1024 threads (max allowed) ✓
                            // • Shared mem: 2×32×32×4 = 8KB ✓
                            // • Larger tiles = more reuse ✓

dim3 gridSize(N/32, N/32);  // 64 × 64 = 4096 blocks

// Resource Check:
// ──────────────
// Registers: Moderate (compiler reports ~40/thread)
//   → 1024 × 40 = 40,960 registers/block
//   → 65536 / 40960 = 1.6 blocks per SM (limited by registers!)
//
// Shared Memory: 8 KB/block
//   → 100KB / 8KB = 12 blocks per SM ✓
//
// Threads: 1024/block
//   → 2048 / 1024 = 2 blocks per SM (limited by threads!)
//
// BOTTLENECK: Threads limit → 2 blocks/SM
// Occupancy = (2 × 1024) / 2048 = 100% ✓
```

### Example C: Histogram

```cpp
__global__ void histogram(unsigned char *image, int *hist, int N) {
    __shared__ int localHist[256];
    // ... (shared memory histogram)
}

// Configuration Analysis:
// ─────────────────────
int N = 1920 * 1080;  // 2M pixels
int blockSize = 256;        // Why 256?
                            // • Need 256 threads to init shared hist ✓
                            // • One thread per histogram bin ✓
                            // • Shared mem: 256 × 4 = 1KB ✓

int gridSize = (N + 255) / 256;  // 8,100 blocks

// Resource Check:
// ──────────────
// Shared Memory: 1 KB/block → 100 blocks/SM possible
// Threads: 256/block → 8 blocks/SM possible
// Bottleneck: Threads → 8 blocks/SM
// Occupancy = (8 × 256) / 2048 = 100% ✓
```

---

## Work Allocation Formulas

### Formula 1: Ceiling Division

```cpp
// Always use ceiling division to avoid missing elements
int gridSize = (N + blockSize - 1) / blockSize;

// Why?
// ────
// N = 1000, blockSize = 256
//
// Wrong: 1000 / 256 = 3 blocks
//        3 × 256 = 768 elements processed
//        232 elements NOT processed! ❌
//
// Correct: (1000 + 255) / 256 = 4 blocks
//          4 × 256 = 1024 thread launches
//          All 1000 elements processed ✓
//          (24 threads check bounds and exit)
```

### Formula 2: 2D Grid Sizing

```cpp
// Image: width × height
dim3 blockSize(TILE_W, TILE_H);
dim3 gridSize((width + TILE_W - 1) / TILE_W,
              (height + TILE_H - 1) / TILE_H);

// Example: 1920×1080 image, 16×16 blocks
gridSize = ((1920 + 15) / 16, (1080 + 15) / 16)
         = (120, 68)
         = 8,160 blocks total
```

### Formula 3: Work Per Thread

```cpp
// When using grid-stride loops
int elementsPerThread = (N + totalThreads - 1) / totalThreads;

where: totalThreads = gridDim.x × blockDim.x

// Example: 10M elements, 256 threads/block, 1024 blocks
totalThreads = 1024 × 256 = 262,144 threads
elementsPerThread = (10000000 + 262143) / 262144 ≈ 38 elements/thread

// Each thread processes ~38 elements in grid-stride loop
```

---

## Performance Guidelines

### Block Size Impact on Performance

```
Benchmark: Vector addition, 10M elements, RTX 3090

╔═══════════════════════════════════════════════════════════════╗
║ Block Size │ Grid Size │ Occupancy │ Time (ms) │ Throughput  ║
╠════════════╪═══════════╪═══════════╪═══════════╪═════════════╣
║     32     │  312,500  │    25%    │   2.1     │   Low       ║
║     64     │  156,250  │    50%    │   1.2     │   Medium    ║
║    128     │   78,125  │    75%    │   0.7     │   Good      ║
║    256     │   39,063  │    88%    │   0.5     │   Excellent ║ ← Best
║    512     │   19,532  │   100%    │   0.5     │   Excellent ║
║   1024     │    9,766  │   100%    │   0.6     │   Good      ║
╚═══════════════════════════════════════════════════════════════╝

Observations:
• < 128: Too low occupancy, poor latency hiding
• 256-512: Sweet spot
• 1024: Occupancy good but less flexible scheduling
```

### Grid Size Impact

```
Benchmark: Same vector addition

╔═══════════════════════════════════════════════════════════════╗
║ Grid Size  │ Description          │ Time (ms) │ Efficiency   ║
╠════════════╪══════════════════════╪═══════════╪══════════════╣
║ 82 × 1     │ numSMs × 1           │   45.0    │ Poor (12 SM  ║
║            │ (too few blocks)     │           │ at a time)   ║
║ ────────────┼──────────────────────┼───────────┼──────────────║
║ 82 × 8     │ numSMs × 8           │    0.7    │ Good         ║
║            │ (grid-stride)        │           │              ║
║ ────────────┼──────────────────────┼───────────┼──────────────║
║ 82 × 16    │ numSMs × 16          │    0.5    │ Excellent    ║
║            │ (optimal)            │           │              ║
║ ────────────┼──────────────────────┼───────────┼──────────────║
║ 39,063     │ Full coverage        │    0.5    │ Excellent    ║
║            │ ((N+255)/256)        │           │              ║
╚═══════════════════════════════════════════════════════════════╝

Key Insight: Need enough blocks to keep all SMs busy!
Minimum: numSMs × (warps per SM / warps per block)
```

---

## Summary: The Golden Rules

```
╔══════════════════════════════════════════════════════════════╗
║              CUDA WORK ALLOCATION GOLDEN RULES               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Block Size:                                                 ║
║  ──────────                                                  ║
║  ✓ Use multiples of 32 (warp size)                           ║
║  ✓ Typical range: 128-512 threads                            ║
║  ✓ Default choice: 256 threads                               ║
║  ✓ 2D problems: 16×16 or 32×32                               ║
║  ✓ 3D problems: 8×8×4 or 4×4×16                              ║
║                                                              ║
║  Grid Size:                                                  ║
║  ─────────                                                   ║
║  ✓ Minimum: numSMs × 2 (keep all SMs busy)                   ║
║  ✓ Optimal: numSMs × 8-16 (good occupancy)                   ║
║  ✓ Use grid-stride for flexibility                           ║
║  ✓ Don't make grid too large (diminishing returns)           ║
║                                                              ║
║  Occupancy:                                                  ║
║  ─────────                                                   ║
║  ✓ Target: 60-75% occupancy                                  ║
║  ✓ Check with: nvcc --ptxas-options=-v                       ║
║  ✓ Profile with: ncu --metrics achieved_occupancy            ║
║  ✓ Higher isn't always better!                               ║
║                                                              ║
║  Resource Management:                                        ║
║  ─────────────────                                           ║
║  ✓ Shared memory: Keep < 48 KB per block                     ║
║  ✓ Registers: Watch for spilling                             ║
║  ✓ Profile before optimizing                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Reference Table

```
╔══════════════════════════════════════════════════════════════════════════╗
║ Problem Type     │ Block Size      │ Grid Size         │ Pattern         ║
╠══════════════════╪═════════════════╪═══════════════════╪═════════════════╣
║ Vector ops       │ 256             │ (N+255)/256       │ 1 elem/thread   ║
║ Matrix ops       │ 16×16 or 32×32  │ (M/16, N/16)      │ Tiled           ║
║ Image process    │ 16×16           │ (W/16, H/16)      │ 2D spatial      ║
║ Reduction        │ 256-512         │ numSMs × 8        │ Tree reduction  ║
║ Histogram        │ 256             │ (N+255)/256       │ Atomic staging  ║
║ Stencil          │ 16×16           │ (W/16, H/16)      │ Halo loading    ║
║ Sort             │ 256-512         │ Data dependent    │ Specialized     ║
║ Graph traversal  │ 256             │ numSMs × 16       │ Frontier-based  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

**For detailed code examples, see existing tutorial files:**
- `04_thread_organization.cu` - Basic thread indexing
- `11_thread_indexing_patterns.md` - 1D/2D/3D patterns
- `15_optimization_case_studies.md` - Performance impact

**Key insight**: Start with defaults (256 threads, numSMs × 8 blocks), then profile and adjust!

