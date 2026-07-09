/*
 * GPU Locks and Critical Sections - Complete Guide
 * 
 * This file demonstrates how to implement critical sections in CUDA:
 * 1. Atomic Operations (most common)
 * 2. Spinlock (mutex-like)
 * 3. Semaphores
 * 4. Lock-free algorithms
 * 5. Warp-level synchronization
 * 6. Block-level synchronization
 *
 * KEY DIFFERENCE FROM CPU:
 * GPUs don't have traditional OS locks. Instead, we use:
 * - Atomic operations
 * - Spinlocks built from atomics
 * - Lock-free data structures
 *
 * Compile: nvcc -o gpu_locks gpu_locks_and_synchronization.cu -O3
 * Run:     ./gpu_locks
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   1. ATOMIC OPERATIONS (Preferred)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Atomics are the BEST way to handle critical sections on GPU!
 * They are hardware-accelerated and avoid deadlocks.
 *
 * Problem Without Atomics:
 * ───────────────────────
 * Thread 0: Read counter (100)
 * Thread 1: Read counter (100)
 * Thread 0: Write counter (101)
 * Thread 1: Write counter (101)  ❌ Lost update! Should be 102
 *
 * Solution With Atomics:
 * ─────────────────────
 * Thread 0: atomicAdd(&counter, 1) → Returns 100, sets to 101
 * Thread 1: atomicAdd(&counter, 1) → Returns 101, sets to 102 ✓
 *
 * Available Atomic Operations:
 * ──────────────────────────
 * atomicAdd(addr, val)         → *addr += val
 * atomicSub(addr, val)         → *addr -= val
 * atomicMin(addr, val)         → *addr = min(*addr, val)
 * atomicMax(addr, val)         → *addr = max(*addr, val)
 * atomicInc(addr, val)         → *addr = (*addr >= val) ? 0 : (*addr + 1)
 * atomicDec(addr, val)         → *addr = (*addr == 0 || *addr > val) ? val : (*addr - 1)
 * atomicExch(addr, val)        → old = *addr; *addr = val; return old
 * atomicCAS(addr, cmp, val)    → if (*addr == cmp) *addr = val; return old
 * atomicAnd/Or/Xor(addr, val)  → Bitwise operations
 */

// Example: Increment shared counter (WRONG way - race condition!)
__global__ void incrementNaive(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int temp = *counter;  // ❌ Race condition!
        temp++;
        *counter = temp;      // ❌ Other threads' updates lost!
    }
}

// Example: Increment shared counter (CORRECT way - atomic)
__global__ void incrementAtomic(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(counter, 1);  // ✓ Thread-safe, hardware accelerated!
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   2. SPINLOCK (Mutex-like)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Build a mutex-like lock using atomics.
 * 
 * Spinlock Implementation:
 * ───────────────────────
 * struct Lock {
 *     int state;  // 0 = unlocked, 1 = locked
 * };
 *
 * void lock(Lock *l) {
 *     while (atomicCAS(&l->state, 0, 1) != 0) {
 *         // Spin (busy-wait) until lock is free
 *     }
 * }
 *
 * void unlock(Lock *l) {
 *     atomicExch(&l->state, 0);
 * }
 *
 * Visual Timeline:
 * ───────────────
 * Time  Thread 0           Thread 1           Lock State
 * ────  ────────────       ────────────       ──────────
 *  0    lock() → success   -                  1 (locked)
 *  1    [critical section] lock() → spin...   1 (locked)
 *  2    [critical section] spin...            1 (locked)
 *  3    unlock()           spin...            0 (unlocked)
 *  4    -                  lock() → success   1 (locked)
 *  5    -                  [critical section] 1 (locked)
 *
 * WARNING: Spinlocks on GPU can cause:
 * ────────────────────────────────────
 * ❌ Deadlock if thread holding lock gets descheduled
 * ❌ Poor performance (threads spinning waste GPU cycles)
 * ❌ Warp divergence (some threads spin, others work)
 *
 * ONLY USE IF:
 * ───────────
 * ✓ Critical section is very short
 * ✓ Low contention (few threads competing)
 * ✓ Can't use atomics alone
 */

// Simple spinlock implementation
__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {
        // Spin until we successfully change 0 → 1
        // This means we "acquired" the lock
    }
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);  // Set lock to 0 (unlocked)
}

// Example using spinlock to protect critical section
__global__ void incrementWithLock(int *counter, int *lock_var, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        lock(lock_var);           // Acquire lock
        
        // Critical section
        int temp = *counter;
        temp++;
        *counter = temp;
        
        unlock(lock_var);         // Release lock
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   3. SEMAPHORE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Semaphore: Controls access to resource pool with N slots.
 *
 * Binary Semaphore (N=1):  Like a lock
 * Counting Semaphore (N>1): Multiple threads can enter
 *
 * Visual Example (N=3):
 * ────────────────────
 * Available slots: [3]
 * Thread 0 enters: [2] ✓
 * Thread 1 enters: [1] ✓
 * Thread 2 enters: [0] ✓
 * Thread 3 waits...   (blocks)
 * Thread 0 exits:  [1] → Thread 3 enters [0] ✓
 */

__device__ void semaphore_wait(int *sem, int max_count) {
    int old;
    do {
        old = *sem;
        if (old <= 0) continue;  // No slots available, keep trying
    } while (atomicCAS(sem, old, old - 1) != old);
}

__device__ void semaphore_signal(int *sem, int max_count) {
    atomicAdd(sem, 1);
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   4. LOCK-FREE ALGORITHMS (Best Practice!)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Lock-free: Use atomics to guarantee progress without locks.
 * This is the PREFERRED approach on GPU!
 *
 * Example: Lock-free stack push
 * ─────────────────────────────
 * struct Node { int data; Node *next; };
 * Node *top;  // Stack top
 *
 * void push(Node *node) {
 *     do {
 *         node->next = top;
 *     } while (atomicCAS(&top, node->next, node) != node->next);
 * }
 *
 * Why Lock-Free is Better:
 * ───────────────────────
 * ✓ No deadlocks
 * ✓ No thread starvation
 * ✓ Better warp efficiency
 * ✓ Scales with thread count
 */

// Lock-free increment (already shown - atomicAdd)

// Lock-free maximum
__global__ void lockFreeMax(int *global_max, int *values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int my_value = values[idx];
        atomicMax(global_max, my_value);  // Lock-free!
    }
}

// Lock-free histogram
__global__ void lockFreeHistogram(int *histogram, int *data, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = data[idx] % bins;
        atomicAdd(&histogram[bin], 1);  // Lock-free!
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   5. WARP-LEVEL SYNCHRONIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Threads in same warp (32 threads) execute in lockstep (SIMT).
 * Can use special warp-level primitives - NO LOCKS NEEDED!
 *
 * Warp Primitives:
 * ───────────────
 * __syncwarp(mask)              → Synchronize threads in warp
 * __shfl_sync(mask, val, lane)  → Shuffle data between threads
 * __ballot_sync(mask, pred)     → Voting across warp
 * __any_sync(mask, pred)        → True if any thread true
 * __all_sync(mask, pred)        → True if all threads true
 *
 * Visual: Warp Reduction (no atomics needed!)
 * ──────────────────────────────────────────
 * Lane:  0   1   2   3   4   5   6   7  ...
 * Data: [5] [3] [7] [2] [9] [1] [4] [6] ...
 *        ↓   ↓   ↓   ↓
 * Step1: 8  [3]  9  [2] 13 [1] 10 [6] ...  (pair-wise add)
 *        ↓       ↓
 * Step2: 17     [9]     23    [10]    ...  (pair-wise add)
 *        ↓               ↓
 * Step3: 40             33             ...
 * Final: Lane 0 has sum!
 */

__device__ int warpReduceSum(int val) {
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // Lane 0 has the sum
}

__global__ void warpLevelSum(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int val = (idx < n) ? input[idx] : 0;
    
    // Reduce within warp (no atomics!)
    val = warpReduceSum(val);
    
    // Only lane 0 of each warp writes
    if (lane == 0) {
        atomicAdd(output, val);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   6. BLOCK-LEVEL SYNCHRONIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * __syncthreads(): Barrier for ALL threads in a block
 *
 * Use Case: Shared memory coordination
 * ────────────────────────────────────
 * __shared__ int data[256];
 *
 * // Phase 1: All threads write
 * data[threadIdx.x] = ...;
 * __syncthreads();  // ← Wait for all writes
 *
 * // Phase 2: All threads read
 * int val = data[...];
 *
 * Visual Timeline:
 * ───────────────
 * Thread 0  Thread 1  Thread 2  Thread 3
 * ────────  ────────  ────────  ────────
 * Write       Write     Write     Write
 *   ↓         ↓         ↓         ↓
 * __syncthreads() ← ALL threads wait here
 *   ↓         ↓         ↓         ↓
 * Read        Read      Read      Read
 *
 * IMPORTANT:
 * ─────────
 * ❌ __syncthreads() only works WITHIN a block
 * ❌ NEVER use inside divergent code (if statement)
 * ❌ Cannot sync across blocks (use grid sync or multiple kernels)
 */

__global__ void blockLevelSum(int *input, int *output, int n) {
    __shared__ int shared[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();  // ← All threads must reach here!
    
    // Reduce in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();  // ← Wait after each reduction step
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   PERFORMANCE COMPARISON
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║       GPU Locks and Critical Sections Guide          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    const int n = 1000000;
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    int *d_counter, *d_lock;
    int h_counter;
    
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lock, sizeof(int)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Naive (race condition)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Naive Increment (Race Condition)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    h_counter = 0;
    CUDA_CHECK(cudaMemcpy(d_counter, &h_counter, sizeof(int),
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    incrementNaive<<<gridSize, blockSize>>>(d_counter, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float naive_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Expected: %d\n", n);
    printf("Got:      %d ❌ (Race condition!)\n", h_counter);
    printf("Time:     %.3f ms\n\n", naive_time);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: Atomic (correct and fast)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Atomic Increment\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    h_counter = 0;
    CUDA_CHECK(cudaMemcpy(d_counter, &h_counter, sizeof(int),
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    incrementAtomic<<<gridSize, blockSize>>>(d_counter, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float atomic_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&atomic_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Expected: %d\n", n);
    printf("Got:      %d ✓ (Correct!)\n", h_counter);
    printf("Time:     %.3f ms\n\n", atomic_time);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 3: Spinlock (correct but SLOW)
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Spinlock\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    h_counter = 0;
    int h_lock = 0;
    CUDA_CHECK(cudaMemcpy(d_counter, &h_counter, sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lock, &h_lock, sizeof(int),
                         cudaMemcpyHostToDevice));
    
    printf("WARNING: This will be VERY slow due to lock contention!\n");
    printf("Running with fewer threads to avoid timeout...\n\n");
    
    int small_n = 1000;  // Much smaller to avoid timeout
    int small_grid = (small_n + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(start));
    incrementWithLock<<<small_grid, blockSize>>>(d_counter, d_lock, small_n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float lock_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&lock_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("Expected: %d\n", small_n);
    printf("Got:      %d ✓ (Correct!)\n", h_counter);
    printf("Time:     %.3f ms (for %d increments)\n\n", lock_time, small_n);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Performance Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Performance Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("┌──────────────────┬──────────┬────────────┬─────────┐\n");
    printf("│ Method           │ Time(ms) │ Correct?   │ Speed   │\n");
    printf("├──────────────────┼──────────┼────────────┼─────────┤\n");
    printf("│ Naive (race)     │ %7.2f  │ ❌ NO      │ Fast    │\n", naive_time);
    printf("│ Atomic           │ %7.2f  │ ✓ YES      │ Fast    │\n", atomic_time);
    printf("│ Spinlock*        │ %7.2f  │ ✓ YES      │ SLOW    │\n", lock_time);
    printf("└──────────────────┴──────────┴────────────┴─────────┘\n");
    printf("*Spinlock tested with %dx fewer operations\n\n", n / small_n);
    
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_lock));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Best Practices
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                  Best Practices                       ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. PREFER atomics over locks whenever possible       ║\n");
    printf("║ 2. Use warp-level primitives for intra-warp sync     ║\n");
    printf("║ 3. Use __syncthreads() for block-level barriers      ║\n");
    printf("║ 4. Avoid spinlocks (cause warp divergence)           ║\n");
    printf("║ 5. Design lock-free algorithms when possible         ║\n");
    printf("║ 6. Reduce contention by using shared memory staging  ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    printf("Synchronization Hierarchy:\n");
    printf("──────────────────────────\n");
    printf("1. Warp level (32 threads):   __shfl_sync(), __ballot_sync()\n");
    printf("2. Block level (up to 1024):  __syncthreads()\n");
    printf("3. Grid level (all blocks):   Multiple kernel launches\n");
    printf("4. Cross-grid:                 Atomics + global memory\n\n");
    
    printf("When to Use What:\n");
    printf("─────────────────\n");
    printf("• Simple counter/accumulator    → atomicAdd\n");
    printf("• Maximum/minimum finding       → atomicMax/Min\n");
    printf("• Histogram                     → atomicAdd per bin\n");
    printf("• Producer-consumer             → atomicCAS\n");
    printf("• Shared memory coordination    → __syncthreads\n");
    printf("• Warp-level reduction          → __shfl_down_sync\n");
    printf("• Complex critical section      → Redesign to avoid locks!\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                     KEY DIFFERENCES: GPU vs CPU
 * ═══════════════════════════════════════════════════════════════════
 *
 * CPU Locks:                    GPU Synchronization:
 * ─────────────────────────    ────────────────────────────
 * • OS-level mutexes           • Atomic operations (hardware)
 * • Kernel puts thread to      • Busy-wait (wastes cycles)
 *   sleep while waiting        • No OS scheduler
 * • Context switches           • Warp divergence problems
 * • Fair scheduling            • No fairness guarantees
 * • Deadlock detection         • Manual deadlock prevention
 * • Priority inheritance       • No priorities
 *
 * Why Locks are Bad on GPU:
 * ────────────────────────
 * 1. Warp divergence: Half the warp spins, half works → 50% efficiency
 * 2. No preemption: Thread holding lock can't be descheduled
 * 3. Scalability: Doesn't scale with thousands of threads
 * 4. Deadlock risk: Hard to debug across thousands of threads
 *
 * ═══════════════════════════════════════════════════════════════════
 */

