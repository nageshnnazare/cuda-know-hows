# CUDA Testing and Debugging Guide

## Comprehensive Guide to Testing, Debugging, and Validation

---

## Table of Contents

1. [Testing Strategies](#testing-strategies)
2. [Unit Testing CUDA Kernels](#unit-testing)
3. [Debugging Tools](#debugging-tools)
4. [Common Bugs and Solutions](#common-bugs)
5. [Performance Testing](#performance-testing)
6. [Continuous Integration](#continuous-integration)
7. [Best Practices](#best-practices)

---

## Testing Strategies {#testing-strategies}

### The Testing Pyramid for CUDA

```
╔════════════════════════════════════════════════════════╗
║                                                        ║
║              ┌──────────────────┐                     ║
║              │  Integration     │  ← Few, slow        ║
║              │  Tests           │                     ║
║         ┌────┴──────────────────┴────┐               ║
║         │  Kernel Tests               │  ← Some      ║
║         │  (GPU specific)             │               ║
║    ┌────┴─────────────────────────────┴────┐         ║
║    │  Unit Tests                            │  ← Many║
║    │  (Host code, utilities)                │         ║
║    └────────────────────────────────────────┘         ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

### Types of Tests

#### **1. Unit Tests**
- Test individual functions
- Fast to run
- Easy to debug
- Run on CPU

#### **2. Kernel Tests**
- Test CUDA kernels
- Verify correctness
- Compare against CPU reference
- Check edge cases

#### **3. Performance Tests**
- Measure throughput
- Compare against baseline
- Detect regressions
- Profile hotspots

#### **4. Integration Tests**
- Test complete workflows
- Multi-kernel interactions
- Multi-GPU scenarios
- Real-world data

---

## Unit Testing CUDA Kernels {#unit-testing}

### Basic Test Structure

```cpp
// test_vector_add.cu
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// The kernel to test
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU reference implementation
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Test function
bool testVectorAdd() {
    const int n = 1000;
    const float tolerance = 1e-5f;
    
    // Allocate host memory
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_c_gpu = (float*)malloc(n * sizeof(float));
    float *h_c_cpu = (float*)malloc(n * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, n * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    
    // Verify results
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(h_c_gpu[i] - h_c_cpu[i]);
        if (diff > tolerance) {
            printf("FAIL at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                   i, h_c_gpu[i], h_c_cpu[i], diff);
            passed = false;
            break;
        }
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    
    return passed;
}

int main() {
    printf("Testing vectorAdd kernel...\n");
    
    if (testVectorAdd()) {
        printf("✓ TEST PASSED\n");
        return 0;
    } else {
        printf("✗ TEST FAILED\n");
        return 1;
    }
}
```

### Test Edge Cases

```cpp
// Test suite for edge cases
void runEdgeCaseTests() {
    printf("Running edge case tests...\n");
    
    // Test 1: Empty input
    testVectorAdd(0);
    
    // Test 2: Single element
    testVectorAdd(1);
    
    // Test 3: Power of 2
    testVectorAdd(1024);
    
    // Test 4: Not power of 2
    testVectorAdd(1000);
    
    // Test 5: Large input
    testVectorAdd(10000000);
    
    // Test 6: Special values
    testSpecialValues();  // NaN, Inf, -Inf, 0
}

void testSpecialValues() {
    float test_values[] = {
        0.0f, -0.0f,           // Zeros
        INFINITY, -INFINITY,    // Infinities
        NAN,                    // Not a number
        FLT_MIN, FLT_MAX,      // Extremes
        1e-38f, 1e38f          // Very small/large
    };
    
    // Test combinations...
}
```

### Parameterized Tests

```cpp
struct TestCase {
    const char *name;
    int size;
    int blockSize;
    bool expectedPass;
};

TestCase testCases[] = {
    {"Small", 100, 32, true},
    {"Medium", 10000, 128, true},
    {"Large", 1000000, 256, true},
    {"Tiny block", 1000, 1, true},
    {"Large block", 1000, 1024, true},
    {"Invalid size", -1, 256, false},
};

void runParameterizedTests() {
    int numTests = sizeof(testCases) / sizeof(TestCase);
    int passed = 0;
    
    for (int i = 0; i < numTests; i++) {
        printf("Test %d: %s... ", i + 1, testCases[i].name);
        
        bool result = testVectorAdd(testCases[i].size, 
                                   testCases[i].blockSize);
        
        if (result == testCases[i].expectedPass) {
            printf("✓ PASS\n");
            passed++;
        } else {
            printf("✗ FAIL\n");
        }
    }
    
    printf("\nResults: %d/%d tests passed\n", passed, numTests);
}
```

---

## Debugging Tools {#debugging-tools}

### 1. cuda-memcheck

**Purpose**: Detect memory errors

```bash
# Basic usage
cuda-memcheck ./my_program

# Check for memory leaks
cuda-memcheck --leak-check full ./my_program

# Check for race conditions
cuda-memcheck --tool racecheck ./my_program

# Check for synchronization errors
cuda-memcheck --tool synccheck ./my_program

# Check for initialization errors
cuda-memcheck --tool initcheck ./my_program
```

**Common Errors Detected**:
```
╔══════════════════════════════════════════════════════════╗
║ Error Type              │ What it Means                  ║
╠═════════════════════════╪════════════════════════════════╣
║ Out-of-bounds access    │ Array index out of range       ║
║ Uninitialized memory    │ Reading before writing         ║
║ Race condition          │ Concurrent unsafe access       ║
║ Memory leak             │ cudaFree() not called          ║
║ Double free             │ cudaFree() called twice        ║
║ Invalid device pointer  │ Using freed/invalid pointer    ║
╚══════════════════════════════════════════════════════════╝
```

### 2. cuda-gdb

**Purpose**: Interactive debugging

```bash
# Start debugger
cuda-gdb ./my_program

# Common commands:
(cuda-gdb) break myKernel        # Set breakpoint
(cuda-gdb) run                   # Run program
(cuda-gdb) cuda thread           # Show current thread
(cuda-gdb) cuda block            # Show current block
(cuda-gdb) cuda kernel           # Show current kernel
(cuda-gdb) print variable        # Print variable value
(cuda-gdb) info cuda threads     # List all threads
(cuda-gdb) cuda thread (0,0,0)   # Switch to thread
(cuda-gdb) step                  # Step one line
(cuda-gdb) continue              # Continue execution
```

**Example Debug Session**:
```gdb
(cuda-gdb) break vectorAdd
Breakpoint 1 at vectorAdd

(cuda-gdb) run
Breakpoint 1, vectorAdd kernel

(cuda-gdb) cuda thread (0,0,0)
Switched to thread (0,0,0)

(cuda-gdb) print idx
$1 = 0

(cuda-gdb) print a[idx]
$2 = 1.0

(cuda-gdb) print b[idx]
$3 = 2.0

(cuda-gdb) next
(cuda-gdb) print c[idx]
$4 = 3.0

(cuda-gdb) continue
```

### 3. printf Debugging

```cpp
// Simple printf in kernel
__global__ void debugKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print from first thread only
    if (idx == 0) {
        printf("Block %d, Thread %d: data[0] = %.2f\n",
               blockIdx.x, threadIdx.x, data[0]);
    }
    
    // Print for specific thread
    if (blockIdx.x == 0 && threadIdx.x < 10) {
        printf("Thread %d: data[%d] = %.2f\n",
               threadIdx.x, idx, data[idx]);
    }
    
    // Conditional debugging
    if (data[idx] < 0) {
        printf("WARNING: Negative value at idx %d: %.2f\n",
               idx, data[idx]);
    }
}
```

**Important**: 
- `printf` has limited buffer (~1MB)
- Flush with `cudaDeviceSynchronize()`
- Can impact performance
- Remove for production

### 4. NVIDIA Nsight Systems

**Purpose**: System-wide profiling

```bash
# Profile application
nsys profile --stats=true ./my_program

# Generate timeline
nsys profile -o timeline ./my_program

# View with GUI
nsys-ui timeline.nsys-rep
```

**What to Look For**:
```
✓ Kernel execution time
✓ Memory transfers (H→D, D→H)
✓ CPU-GPU synchronization
✓ Stream utilization
✓ API overhead
✓ Gaps (idle time)
```

### 5. NVIDIA Nsight Compute

**Purpose**: Kernel-level profiling

```bash
# Profile specific kernel
ncu --kernel-name myKernel ./my_program

# Get all metrics
ncu --set full ./my_program

# Interactive mode
ncu-ui
```

**Key Metrics**:
```
╔═══════════════════════════════════════════════════════════╗
║ Metric                  │ What to Check                   ║
╠═════════════════════════╪═════════════════════════════════╣
║ SM Efficiency           │ > 60% is good                   ║
║ Occupancy               │ > 50% usually sufficient        ║
║ Memory Throughput       │ Compare to peak bandwidth       ║
║ Warp Execution          │ Look for divergence             ║
║ Bank Conflicts          │ Should be minimal               ║
║ Cache Hit Rate          │ Higher is better                ║
║ Register Usage          │ Can limit occupancy             ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Common Bugs and Solutions {#common-bugs}

### Bug 1: Race Condition

```cpp
// ❌ WRONG: Race condition
__global__ void incrementCounter(int *counter) {
    int temp = *counter;
    temp++;
    *counter = temp;  // Multiple threads overwrite each other!
}

// ✓ CORRECT: Use atomics
__global__ void incrementCounter(int *counter) {
    atomicAdd(counter, 1);  // Thread-safe
}
```

### Bug 2: Off-by-One Error

```cpp
// ❌ WRONG: Buffer overflow
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 0;  // Might access beyond array!
}

// ✓ CORRECT: Bounds checking
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // Check bounds
        data[idx] = 0;
    }
}
```

### Bug 3: Uninitialized Memory

```cpp
// ❌ WRONG: Reading uninitialized memory
float *d_data;
cudaMalloc(&d_data, n * sizeof(float));
// Use d_data without initialization

// ✓ CORRECT: Initialize
float *d_data;
cudaMalloc(&d_data, n * sizeof(float));
cudaMemset(d_data, 0, n * sizeof(float));  // Initialize to 0
// or
cudaMemcpy(d_data, h_data, n * sizeof(float), 
           cudaMemcpyHostToDevice);
```

### Bug 4: Deadlock with __syncthreads()

```cpp
// ❌ WRONG: Conditional synchronization
__global__ void kernel(int *data) {
    if (threadIdx.x < 64) {
        __syncthreads();  // Only some threads reach here!
    }
}

// ✓ CORRECT: Unconditional barrier
__global__ void kernel(int *data) {
    __syncthreads();  // All threads reach here
    if (threadIdx.x < 64) {
        // Work
    }
}
```

### Bug 5: Memory Leak

```cpp
// ❌ WRONG: Memory leak
void leakyFunction() {
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    // Use d_data
    // Missing cudaFree!
}

// ✓ CORRECT: Always free
void goodFunction() {
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    // Use d_data
    cudaFree(d_data);  // Clean up
}
```

### Bug 6: Incorrect Grid Size

```cpp
// ❌ WRONG: Integer division loses remainder
int gridSize = n / blockSize;  // Loses elements if n % blockSize != 0

// ✓ CORRECT: Ceiling division
int gridSize = (n + blockSize - 1) / blockSize;
```

### Bug 7: Host-Device Pointer Confusion

```cpp
// ❌ WRONG: Using device pointer on host
float *d_data;
cudaMalloc(&d_data, n * sizeof(float));
printf("%.2f\n", d_data[0]);  // Segmentation fault!

// ✓ CORRECT: Copy to host first
float *d_data;
cudaMalloc(&d_data, n * sizeof(float));
float h_value;
cudaMemcpy(&h_value, d_data, sizeof(float), 
           cudaMemcpyDeviceToHost);
printf("%.2f\n", h_value);
```

---

## Performance Testing {#performance-testing}

### Benchmark Template

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

class CUDATimer {
private:
    cudaEvent_t start, stop;
    
public:
    CUDATimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Benchmark function
void benchmarkKernel(int n, int iterations) {
    // Setup
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    CUDATimer timer;
    
    // Warmup
    myKernel<<<gridSize, blockSize>>>(d_data, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.startTimer();
    for (int i = 0; i < iterations; i++) {
        myKernel<<<gridSize, blockSize>>>(d_data, n);
    }
    float totalTime = timer.stopTimer();
    
    // Report
    float avgTime = totalTime / iterations;
    float throughput = (n * sizeof(float) * 2) / (avgTime * 1e6);
    
    printf("Average time: %.3f ms\n", avgTime);
    printf("Throughput: %.2f GB/s\n", throughput);
    
    // Cleanup
    cudaFree(d_data);
}
```

### Regression Testing

```cpp
struct PerformanceBaseline {
    const char *name;
    float baseline_ms;
    float tolerance;  // Allow 10% variance
};

PerformanceBaseline baselines[] = {
    {"vectorAdd_1M", 0.5f, 0.1f},
    {"matmul_1024", 2.0f, 0.1f},
    {"reduction_10M", 1.0f, 0.1f},
};

bool checkPerformanceRegression() {
    bool allPassed = true;
    
    for (auto &baseline : baselines) {
        float actual = measurePerformance(baseline.name);
        float expected = baseline.baseline_ms;
        float threshold = expected * (1 + baseline.tolerance);
        
        if (actual > threshold) {
            printf("REGRESSION: %s\n", baseline.name);
            printf("  Expected: %.3f ms\n", expected);
            printf("  Actual: %.3f ms\n", actual);
            printf("  Slowdown: %.1f%%\n", 
                   100.0 * (actual - expected) / expected);
            allPassed = false;
        } else {
            printf("✓ %s: %.3f ms\n", baseline.name, actual);
        }
    }
    
    return allPassed;
}
```

---

## Continuous Integration {#continuous-integration}

### GitHub Actions Example

```yaml
# .github/workflows/cuda-tests.yml
name: CUDA Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:11.8.0-devel-ubuntu22.04
      
    steps:
    - uses: actions/checkout@v2
    
    - name: Build tests
      run: |
        make tests
        
    - name: Run unit tests
      run: |
        ./run_tests
        
    - name: Run cuda-memcheck
      run: |
        cuda-memcheck ./run_tests
        
    - name: Check performance
      run: |
        ./benchmark --regression
```

---

## Best Practices {#best-practices}

### Testing Checklist

```
□ Test with small inputs first
□ Test edge cases (0, 1, max)
□ Test with special values (NaN, Inf)
□ Compare against CPU reference
□ Use appropriate tolerance for floats
□ Test with different block sizes
□ Test with different grid sizes
□ Run cuda-memcheck regularly
□ Profile before optimizing
□ Document expected performance
□ Automate regression tests
□ Test on multiple GPUs
□ Test with different compute capabilities
```

### Debugging Workflow

```
1. Reproduce the bug consistently
2. Minimize test case
3. Add printf debugging
4. Use cuda-memcheck
5. Use cuda-gdb if needed
6. Fix the bug
7. Add test to prevent regression
8. Document the issue
```

---

## Summary

```
╔══════════════════════════════════════════════════════════════╗
║                     Testing & Debugging                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Tools:                                                      ║
║  ✓ cuda-memcheck    → Memory errors                         ║
║  ✓ cuda-gdb         → Interactive debugging                 ║
║  ✓ printf          → Simple debugging                       ║
║  ✓ Nsight Systems   → System profiling                      ║
║  ✓ Nsight Compute   → Kernel profiling                      ║
║                                                              ║
║  Best Practices:                                             ║
║  ✓ Write unit tests                                         ║
║  ✓ Test edge cases                                          ║
║  ✓ Compare vs CPU                                           ║
║  ✓ Check for errors after every CUDA call                   ║
║  ✓ Use proper tolerances for float comparison               ║
║  ✓ Profile before optimizing                                ║
║  ✓ Automate regression testing                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

**Remember**: Good testing catches bugs early. Good debugging finds them quickly. Together, they make CUDA development productive!

