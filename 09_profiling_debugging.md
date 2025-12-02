# CUDA Tutorial - Part 9: Profiling and Debugging Tools

## Table of Contents
1. [Overview](#overview)
2. [CUDA Error Checking](#cuda-error-checking)
3. [CUDA-MEMCHECK](#cuda-memcheck)
4. [CUDA-GDB Debugger](#cuda-gdb-debugger)
5. [NVIDIA Nsight Systems](#nvidia-nsight-systems)
6. [NVIDIA Nsight Compute](#nvidia-nsight-compute)
7. [Legacy nvprof](#legacy-nvprof)
8. [Performance Metrics](#performance-metrics)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Best Practices](#best-practices)

---

## Overview

Profiling and debugging are essential skills for CUDA development. This guide covers the complete toolkit available for analyzing and optimizing CUDA applications.

### The CUDA Debugging & Profiling Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. DEVELOPMENT                                               │
│     ┌──────────────┐                                         │
│     │ Write Code   │                                         │
│     └──────┬───────┘                                         │
│            │                                                  │
│  2. ERROR CHECKING                                            │
│     ┌──────▼───────┐                                         │
│     │ CUDA_CHECK() │  Runtime error detection                │
│     │ assert()     │  Device-side assertions                 │
│     └──────┬───────┘                                         │
│            │                                                  │
│  3. MEMORY DEBUGGING                                          │
│     ┌──────▼────────┐                                        │
│     │ cuda-memcheck │  Memory errors, race conditions        │
│     └──────┬────────┘                                        │
│            │                                                  │
│  4. FUNCTIONAL DEBUGGING                                      │
│     ┌──────▼───────┐                                         │
│     │  cuda-gdb    │  Breakpoints, variable inspection       │
│     └──────┬───────┘                                         │
│            │                                                  │
│  5. PERFORMANCE PROFILING                                     │
│     ┌──────▼──────────┐    ┌──────────────────┐             │
│     │ Nsight Systems  │    │ Nsight Compute   │             │
│     │ (Timeline view) │    │ (Kernel details) │             │
│     └─────────────────┘    └──────────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## CUDA Error Checking

### Why Error Checking is Critical

CUDA kernel launches are **asynchronous** by default. Errors may not appear immediately!

```c
// BAD: Silent failure
myKernel<<<blocks, threads>>>(args);
// Kernel might have failed, but we don't know!

// GOOD: Immediate error checking
myKernel<<<blocks, threads>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
}
```

### Comprehensive Error Checking Macro

```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "Error: %s (%d)\n", \
                    cudaGetErrorString(err), err); \
            fprintf(stderr, "Error name: %s\n", cudaGetErrorName(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

### Kernel Launch Error Checking

```c
// Method 1: Check after kernel launch
myKernel<<<blocks, threads>>>(args);
CUDA_CHECK(cudaGetLastError());        // Check launch errors
CUDA_CHECK(cudaDeviceSynchronize());   // Check execution errors

// Method 2: Using CUDA_LAUNCH_BLOCKING
// Set environment variable before running:
// export CUDA_LAUNCH_BLOCKING=1
// This makes all kernel launches synchronous (easier debugging)
```

### Device-Side Assertions

```c
__global__ void kernelWithAssert(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Device-side assertion
    assert(idx < n && "Index out of bounds!");
    
    if (idx < n) {
        assert(data[idx] >= 0.0f && "Data must be non-negative");
        data[idx] = sqrtf(data[idx]);
    }
}

// Compile with: nvcc -G (enables device-side assertions)
```

### Common CUDA Errors

| Error Code | Name | Common Cause |
|------------|------|--------------|
| 1 | cudaErrorInvalidValue | Invalid parameter to API call |
| 2 | cudaErrorMemoryAllocation | Out of memory |
| 4 | cudaErrorInitializationError | Driver/runtime initialization failed |
| 11 | cudaErrorInvalidDevice | Invalid device ordinal |
| 30 | cudaErrorUnknown | Unknown error (often indicates GPU hang) |
| 77 | cudaErrorIllegalAddress | Invalid memory access |
| 98 | cudaErrorInvalidSource | Invalid PTX/cubin |
| 719 | cudaErrorLaunchTimeout | Kernel exceeded timeout (Windows TDR) |

---

## CUDA-MEMCHECK

**cuda-memcheck** is a suite of tools for detecting memory errors and race conditions.

### Available Tools

```
┌────────────────────────────────────────────────────────────┐
│              CUDA-MEMCHECK TOOL SUITE                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. MEMCHECK (default)                                     │
│     • Out-of-bounds memory accesses                        │
│     • Misaligned memory accesses                           │
│     • Memory leaks                                         │
│                                                             │
│  2. RACECHECK                                              │
│     • Shared memory race conditions                        │
│     • Global memory race conditions                        │
│     • Missing __syncthreads()                              │
│                                                             │
│  3. INITCHECK                                              │
│     • Uninitialized device memory reads                    │
│     • Uninitialized shared memory reads                    │
│                                                             │
│  4. SYNCCHECK                                              │
│     • Invalid synchronization usage                        │
│     • Deadlocks                                            │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Basic Usage

```bash
# Default memcheck
cuda-memcheck ./my_program

# Specific tool
cuda-memcheck --tool racecheck ./my_program
cuda-memcheck --tool initcheck ./my_program
cuda-memcheck --tool synccheck ./my_program

# With leak checking
cuda-memcheck --leak-check full ./my_program

# Save report to file
cuda-memcheck --log-file report.txt ./my_program
```

### Example: Detecting Out-of-Bounds Access

**Code with bug:**
```c
__global__ void buggyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: No boundary check!
    data[idx] = idx * 2.0f;  // Writes beyond array bounds
}

int main() {
    int N = 100;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Launch with more threads than data size
    buggyKernel<<<10, 32>>>(d_data, N);  // 320 threads, only 100 elements!
    
    cudaFree(d_data);
    return 0;
}
```

**Running cuda-memcheck:**
```bash
$ cuda-memcheck ./buggy_program

========= CUDA-MEMCHECK
========= Invalid __global__ write of size 4
=========     at 0x00000148 in buggyKernel(float*, int)
=========     by thread (0,0,0) in block (4,0,0)
=========     Address 0x7f8e44000190 is out of bounds
=========
========= ERROR SUMMARY: 220 errors
```

### Example: Detecting Race Conditions

**Code with race condition:**
```c
__global__ void racyKernel(float *data) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    
    shared[tid] = data[tid];
    // BUG: Missing __syncthreads()!
    
    // Another thread might not have written yet!
    if (tid < 255) {
        data[tid] = shared[tid] + shared[tid + 1];  // RACE!
    }
}
```

**Running racecheck:**
```bash
$ cuda-memcheck --tool racecheck ./racy_program

========= RACECHECK
========= Shared memory race hazard detected
=========     Thread 1 read from 0x00000004 in block (0,0,0)
=========     Thread 0 wrote to 0x00000004 in block (0,0,0)
=========     Potential data race on shared memory
```

### Memory Leak Detection

```bash
# Compile with line number info
nvcc -lineinfo -o program program.cu

# Run with leak checking
cuda-memcheck --leak-check full ./program

# Output shows:
# - Leaked memory size
# - Allocation location
# - Stack trace
```

---

## CUDA-GDB Debugger

**cuda-gdb** is an extension of GNU GDB for debugging CUDA applications.

### Compiling for Debugging

```bash
# Enable debug info (-g) and disable optimization (-G)
nvcc -g -G -o program program.cu

# Without -G, many variables will be optimized away
```

### Basic cuda-gdb Commands

```bash
# Start debugger
cuda-gdb ./program

# Common commands:
(cuda-gdb) break main                    # Set breakpoint at function
(cuda-gdb) break kernel<<<*,*>>>()      # Break at kernel launch
(cuda-gdb) break myKernel               # Break in device function
(cuda-gdb) run                          # Run program
(cuda-gdb) continue                     # Continue execution
(cuda-gdb) step                         # Step into
(cuda-gdb) next                         # Step over
(cuda-gdb) print var                    # Print variable
(cuda-gdb) info cuda threads            # Show CUDA threads
(cuda-gdb) info cuda blocks             # Show CUDA blocks
(cuda-gdb) cuda thread (0,0,0)          # Switch to specific thread
(cuda-gdb) cuda block (0,0,0)           # Switch to specific block
```

### CUDA-Specific Commands

```
┌──────────────────────────────────────────────────────────────┐
│                   CUDA-GDB SPECIAL COMMANDS                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ info cuda devices          List all CUDA devices             │
│ info cuda sms              List streaming multiprocessors    │
│ info cuda warps            List warps                        │
│ info cuda lanes            List lanes in warp                │
│ info cuda kernels          List active kernels               │
│ info cuda blocks           List blocks                       │
│ info cuda threads          List threads                      │
│                                                               │
│ cuda device <n>            Switch to device n                │
│ cuda sm <n>                Switch to SM n                    │
│ cuda warp <n>              Switch to warp n                  │
│ cuda lane <n>              Switch to lane n                  │
│ cuda kernel <n>            Switch to kernel n                │
│ cuda block <x,y,z>         Switch to block (x,y,z)           │
│ cuda thread <x,y,z>        Switch to thread (x,y,z)          │
│                                                               │
│ cuda thread blockIdx       Show current block index          │
│ cuda thread threadIdx      Show current thread index         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Example Debugging Session

```bash
$ cuda-gdb ./program

(cuda-gdb) break myKernel
Breakpoint 1 at 0x4012a0: file program.cu, line 25.

(cuda-gdb) run
Starting program...
[Switching to CUDA Kernel 0, Grid 1, Block (0,0,0), Thread (0,0,0)]

Breakpoint 1, myKernel (data=0x7fff12000000, n=1024)
    at program.cu:25
25          int idx = blockIdx.x * blockDim.x + threadIdx.x;

(cuda-gdb) print blockIdx
$1 = {x = 0, y = 0, z = 0}

(cuda-gdb) print threadIdx
$2 = {x = 0, y = 0, z = 0}

(cuda-gdb) print blockDim
$3 = {x = 256, y = 1, z = 1}

(cuda-gdb) cuda thread (1,0,0)
[Switching to CUDA Kernel 0, Grid 1, Block (0,0,0), Thread (1,0,0)]

(cuda-gdb) print idx
$4 = 1

(cuda-gdb) print data[idx]
$5 = 2.5

(cuda-gdb) continue
```

### Conditional Breakpoints

```bash
# Break when specific thread reaches location
(cuda-gdb) break myKernel if threadIdx.x == 5 && blockIdx.x == 2

# Break when variable has specific value
(cuda-gdb) break myKernel if idx == 100

# Break in specific block
(cuda-gdb) cuda block (2,1,0)
(cuda-gdb) break myKernel
```

---

## NVIDIA Nsight Systems

**Nsight Systems** provides system-wide performance analysis and timeline visualization.

### What Nsight Systems Shows

```
┌────────────────────────────────────────────────────────────────┐
│                 NSIGHT SYSTEMS TIMELINE VIEW                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPU Thread 0: [████Main████]                                  │
│                                                                 │
│  CUDA Context:                                                  │
│    Compute:    [═══Kernel1═══]  [═══Kernel2═══]  [═══K3═══]   │
│    MemcpyHtoD: [▓▓▓]            [▓▓▓]                          │
│    MemcpyDtoH:          [▓▓▓]            [▓▓▓]                 │
│                                                                 │
│  GPU Utilization: ████████░░░░██████░░░░░░████                │
│                                                                 │
│  Memory Transfer: ▓▓▓░░░░░▓▓▓░░░░░░░░▓▓▓                      │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│  Key Insights:                                                  │
│  • Overlapping of kernels and transfers                        │
│  • Idle time (optimization opportunities)                      │
│  • CPU-GPU synchronization points                              │
│  • Stream concurrency                                          │
└────────────────────────────────────────────────────────────────┘
```

### Basic Usage

```bash
# Profile application
nsys profile -o report ./my_program

# Profile with specific options
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report \
    --force-overwrite=true \
    ./my_program

# View report in GUI
nsys-ui report.nsys-rep

# Generate text report
nsys stats report.nsys-rep
```

### Command-Line Options

```bash
# Trace CUDA API calls
nsys profile --trace=cuda ./program

# Trace OpenMP, MPI
nsys profile --trace=cuda,openmp,mpi ./program

# Sample CPU stacks
nsys profile --sample=cpu ./program

# Set duration
nsys profile --duration=10 ./program

# Delay start
nsys profile --delay=2 ./program

# Profile only specific time range
nsys profile --capture-range=cudaProfilerApi ./program
```

### NVTX Markers for Custom Ranges

NVTX (NVIDIA Tools Extension) allows you to annotate your code:

```c
#include <nvToolsExt.h>

void myFunction() {
    // Mark range
    nvtxRangePush("MyFunction");
    
    // Your code here
    
    nvtxRangePop();
}

// Colored markers
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF00FF00;  // Green
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = "Data Processing";

nvtxRangePushEx(&eventAttrib);
// ... processing ...
nvtxRangePop();

// Compile with: -lnvToolsExt
```

### Interpreting Results

**Key Metrics to Look For:**

1. **Kernel Execution Time**
   - Long kernels may need optimization
   - Very short kernels → launch overhead dominates

2. **Memory Transfer Time**
   - Large transfers indicate bottleneck
   - Check if transfers can be overlapped

3. **Idle Time**
   - GPU idle → not enough work
   - CPU waiting → synchronization issue

4. **Stream Concurrency**
   - Multiple streams should show overlap
   - Sequential execution indicates dependencies

---

## NVIDIA Nsight Compute

**Nsight Compute** provides detailed kernel-level performance analysis.

### What Nsight Compute Analyzes

```
┌────────────────────────────────────────────────────────────────┐
│              NSIGHT COMPUTE KERNEL ANALYSIS                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PERFORMANCE METRICS:                                           │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ Compute (SM) Throughput:         78% ████████████░░  │      │
│  │ Memory Throughput:               92% ████████████░   │      │
│  │ Achieved Occupancy:              45% ████████░░░░░░  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  LIMITERS:                                                      │
│  • Memory bandwidth (primary bottleneck)                        │
│  • Uncoalesced global memory accesses                          │
│  • Low warp occupancy                                          │
│                                                                 │
│  RECOMMENDATIONS:                                               │
│  ✓ Use shared memory to reduce global accesses                │
│  ✓ Improve memory coalescing                                   │
│  ✓ Increase threads per block to improve occupancy            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Basic Usage

```bash
# Profile single kernel launch
ncu ./my_program

# Profile all kernel launches
ncu --kernel-name-base function ./my_program

# Profile specific kernel by regex
ncu --kernel-name myKernel ./my_program

# Full metric collection
ncu --set full -o report ./my_program

# View report in GUI
ncu-ui report.ncu-rep
```

### Important Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Achieved Occupancy** | Active warps / max warps | 50-100% |
| **SM Efficiency** | SM utilization | 80-100% |
| **Memory Throughput** | Memory bandwidth used | 60-100% |
| **Warp Execution Efficiency** | Non-divergent warps | 90-100% |
| **Global Load/Store Efficiency** | Coalesced accesses | 80-100% |
| **Shared Memory Conflicts** | Bank conflicts | 0-5% |
| **Branch Efficiency** | Non-divergent branches | 90-100% |

### Metric Groups

```bash
# Memory workload analysis
ncu --set memory ./program

# Compute workload analysis  
ncu --set compute ./program

# Launch statistics
ncu --set launch ./program

# Instruction statistics
ncu --set instruction ./program

# Custom metrics
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld ./program
```

### Baseline Comparison

```bash
# Create baseline
ncu --set full -o baseline ./program

# Compare with optimized version
ncu --set full -o optimized ./program_v2

# View comparison
ncu-ui baseline.ncu-rep optimized.ncu-rep
```

### Roofline Analysis

Nsight Compute provides roofline model visualization:

```
    Performance
        ↑
 Peak   │         ╱────────────  Compute Bound
        │       ╱
        │     ╱
        │   ╱  ← Your Kernel
        │ ╱
        │╱──────────────────────  Memory Bound
        └──────────────────────────────→
              Arithmetic Intensity
```

---

## Legacy nvprof

**Note:** nvprof is deprecated but still useful for older GPUs.

### Basic Usage

```bash
# Basic profiling
nvprof ./my_program

# Detailed metrics
nvprof --metrics all ./my_program

# Timeline trace
nvprof --print-gpu-trace ./my_program

# Export for visual profiler
nvprof --export-profile timeline.prof ./my_program
```

### Useful Metrics

```bash
# Measure achieved occupancy
nvprof --metrics achieved_occupancy ./program

# Memory efficiency
nvprof --metrics gld_efficiency,gst_efficiency ./program

# Branch efficiency
nvprof --metrics branch_efficiency ./program

# Warp execution efficiency
nvprof --metrics warp_execution_efficiency ./program

# Memory throughput
nvprof --metrics dram_read_throughput,dram_write_throughput ./program
```

---

## Performance Metrics

### Understanding Key Performance Indicators

#### 1. Occupancy

**Definition:** Ratio of active warps to maximum possible warps per SM.

```
Occupancy = Active Warps / Maximum Warps

Maximum Warps per SM = min(
    MaxWarpsPerSM,
    RegisterLimit / RegistersPerThread,
    SharedMemLimit / SharedMemPerBlock
)
```

**Factors Affecting Occupancy:**
- Threads per block
- Registers per thread
- Shared memory per block
- Block size

**Example Calculation:**
```
GPU: RTX 3080
- Max warps per SM: 64
- Threads per block: 256 (8 warps)
- Registers per thread: 32
- Register file: 65536 registers per SM

Register limit: 65536 / 32 = 2048 threads = 64 warps
Blocks per SM: 64 / 8 = 8 blocks

Occupancy: 8 warps × 8 blocks = 64 warps = 100% ✓
```

#### 2. Memory Bandwidth Utilization

```
Achieved Bandwidth = (Bytes Transferred) / (Time)
Efficiency = (Achieved / Peak) × 100%
```

**Example:**
```
Transfer: 1 GB
Time: 10 ms
Achieved: 100 GB/s
Peak (RTX 3080): 760 GB/s
Efficiency: 13.2% ← Room for improvement!
```

#### 3. Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Accessed

High intensity → Compute bound (good for GPU)
Low intensity  → Memory bound (optimize memory)
```

#### 4. IPC (Instructions Per Cycle)

```
IPC = Instructions Executed / SM Cycles

Higher IPC = Better utilization
Low IPC → Check for:
  - Memory stalls
  - Branch divergence
  - Insufficient parallelism
```

---

## Common Issues and Solutions

### Issue 1: Illegal Memory Access

**Symptom:**
```
cuda-memcheck: Invalid __global__ write of size 4
```

**Common Causes:**
- Array index out of bounds
- Incorrect grid/block dimensions
- Pointer arithmetic error

**Solution:**
```c
__global__ void safeKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Always check bounds!
    if (idx < n) {
        data[idx] = 0.0f;
    }
}
```

### Issue 2: Unspecified Launch Failure

**Symptom:**
```
cudaError: unspecified launch failure
```

**Common Causes:**
- Kernel crash (division by zero, assert failure)
- Stack overflow
- Invalid memory access
- Timeout (Windows TDR)

**Debug Steps:**
```bash
# 1. Run with cuda-memcheck
cuda-memcheck ./program

# 2. Enable launch blocking
export CUDA_LAUNCH_BLOCKING=1
./program

# 3. Compile with debug info and use cuda-gdb
nvcc -g -G -o program program.cu
cuda-gdb ./program
```

### Issue 3: Low Occupancy

**Symptom:**
```
Achieved Occupancy: 25%
```

**Common Causes:**
- Too many registers per thread
- Too much shared memory
- Block size too small

**Solutions:**
```bash
# Check resource usage
nvcc --ptxas-options=-v program.cu

# Output shows:
# ptxas info : Used 48 registers, 1024 bytes smem

# Limit register usage
nvcc --maxrregcount=32 program.cu

# Increase block size
dim3 block(256);  // Instead of 64
```

### Issue 4: Poor Memory Coalescing

**Symptom:**
```
Global Load Efficiency: 25%
```

**Problem:**
```c
// Bad: Strided access
__global__ void bad(float *data, int stride) {
    int idx = threadIdx.x * stride;  // Non-coalesced!
    data[idx] = 0.0f;
}
```

**Solution:**
```c
// Good: Sequential access
__global__ void good(float *data) {
    int idx = threadIdx.x;  // Coalesced!
    data[idx] = 0.0f;
}
```

### Issue 5: Bank Conflicts

**Symptom:**
```
Shared Memory Bank Conflicts: High
```

**Problem:**
```c
__shared__ float shared[32][32];
float val = shared[threadIdx.x][0];  // All threads access column 0!
```

**Solution:**
```c
__shared__ float shared[32][33];  // Padding avoids conflicts
float val = shared[threadIdx.x][0];  // Now conflict-free
```

---

## Best Practices

### Development Workflow

```
1. Write correct code first
   ↓
2. Add comprehensive error checking
   ↓
3. Test with cuda-memcheck
   ↓
4. Profile with Nsight Systems (find bottlenecks)
   ↓
5. Optimize hot kernels with Nsight Compute
   ↓
6. Iterate and measure
```

### Error Checking Checklist

- [ ] Check all CUDA API return values
- [ ] Check kernel launch errors with `cudaGetLastError()`
- [ ] Use `cudaDeviceSynchronize()` for execution errors
- [ ] Add device assertions in debug builds
- [ ] Test with `CUDA_LAUNCH_BLOCKING=1`
- [ ] Run cuda-memcheck before release

### Profiling Checklist

- [ ] Profile with release build (`-O3`, no `-G`)
- [ ] Use representative input data
- [ ] Run multiple iterations for stable results
- [ ] Profile entire application, not just kernels
- [ ] Check CPU-GPU transfer overhead
- [ ] Verify occupancy and memory efficiency
- [ ] Look for optimization opportunities in timeline

### Performance Optimization Steps

```
┌─────────────────────────────────────────────────────────┐
│           OPTIMIZATION PRIORITY ORDER                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Algorithm Selection (10-1000x speedup)              │
│     Choose the right algorithm first!                   │
│                                                          │
│  2. Memory Access Optimization (2-10x)                  │
│     • Coalescing                                        │
│     • Shared memory                                     │
│     • Reduce transfers                                  │
│                                                          │
│  3. Occupancy Optimization (1.5-3x)                     │
│     • Adjust block size                                 │
│     • Reduce register usage                             │
│     • Optimize shared memory                            │
│                                                          │
│  4. Instruction Optimization (1.2-2x)                   │
│     • Fast math                                         │
│     • Unroll loops                                      │
│     • Reduce divergence                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Profiling Tools Quick Reference

| Task | Tool | Command |
|------|------|---------|
| **Memory errors** | cuda-memcheck | `cuda-memcheck ./program` |
| **Race conditions** | racecheck | `cuda-memcheck --tool racecheck ./program` |
| **Debugging** | cuda-gdb | `cuda-gdb ./program` |
| **System timeline** | Nsight Systems | `nsys profile -o report ./program` |
| **Kernel analysis** | Nsight Compute | `ncu -o report ./program` |
| **Quick metrics** | nvprof (legacy) | `nvprof --metrics all ./program` |

---

## Summary

### Essential Commands for Every Developer

```bash
# 1. Always compile with error checking
nvcc -o program program.cu

# 2. Test for memory errors
cuda-memcheck ./program

# 3. Profile system-wide
nsys profile -o timeline ./program

# 4. Analyze critical kernels
ncu --set full -o kernel_analysis ./program

# 5. Debug with cuda-gdb when needed
cuda-gdb ./program
```

### Key Takeaways

✓ **Always check for errors** - CUDA errors are silent by default  
✓ **Use cuda-memcheck regularly** - Catches memory bugs early  
✓ **Profile before optimizing** - Measure, don't guess  
✓ **Start with Nsight Systems** - Find bottlenecks first  
✓ **Use Nsight Compute for details** - Optimize hot kernels  
✓ **Iterate and measure** - Verify each optimization helps  

---

## Additional Resources

### Official Documentation

- [CUDA-MEMCHECK Documentation](https://docs.nvidia.com/cuda/cuda-memcheck/)
- [CUDA-GDB User Guide](https://docs.nvidia.com/cuda/cuda-gdb/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Tutorials and Guides

- [NVIDIA Developer Blog: Profiling](https://developer.nvidia.com/blog/tag/profiling/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Performance Analysis](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)

### Training

- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/)

---

**Next Steps:** Return to the [main README](README.md) to continue the tutorial series!

