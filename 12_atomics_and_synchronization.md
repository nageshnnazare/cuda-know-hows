# 12 — Atomics & Synchronization

> Part of **[CUDA Know-Hows](README.md)**. Prev: [11 — Matrix multiplication](11_matrix_multiplication.md).
> Next: [13 — Streams & concurrency](13_streams_and_concurrency.md).
> Runnable code: [`examples/gpu_locks_and_synchronization.cu`](examples/gpu_locks_and_synchronization.cu),
> [`examples/08_advanced_topics.cu`](examples/08_advanced_topics.cu). The complete
> guide to atomics, locks/critical sections, barriers, and lock-free patterns on
> the GPU.

## Understanding Synchronization in CUDA Programming

---

## Table of Contents

1. [Introduction: Why GPU Locks Are Different](#introduction)
2. [Atomic Operations - The Preferred Approach](#atomic-operations)
3. [Spinlocks - When and Why to Avoid Them](#spinlocks)
4. [Semaphores in GPU Programming](#semaphores)
5. [Lock-Free Algorithms](#lock-free-algorithms)
6. [Warp-Level Synchronization](#warp-level-synchronization)
7. [Block-Level Synchronization](#block-level-synchronization)
8. [Grid-Level Synchronization](#grid-level-synchronization)
9. [Performance Comparison](#performance-comparison)
10. [Best Practices and Guidelines](#best-practices)
11. [Common Patterns and Use Cases](#common-patterns)
12. [Debugging and Troubleshooting](#debugging)

---

## Introduction: Why GPU Locks Are Different {#introduction}

### The Fundamental Problem

When thousands of GPU threads try to access shared data simultaneously, we face the classic **race condition** problem. However, GPUs handle this very differently from CPUs.

### CPU vs GPU: A Tale of Two Architectures

```
╔══════════════════════════════════════════════════════════════════╗
║                    CPU SYNCHRONIZATION                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Thread 1: mutex.lock()    → OS puts thread to sleep             ║
║  Thread 2: mutex.lock()    → Gets lock immediately               ║
║            ...critical section...                                ║
║            mutex.unlock()  → OS wakes Thread 1                   ║
║                                                                  ║
║  Features:                                                       ║
║  ✓ OS-level scheduler                                            ║
║  ✓ Context switching                                             ║
║  ✓ Fair scheduling (FIFO, priority-based)                        ║
║  ✓ Deadlock detection                                            ║
║  ✓ Priority inheritance                                          ║
║  ✓ Thread sleeps (saves CPU cycles)                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                    GPU SYNCHRONIZATION                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Thread 1: atomicCAS(&lock, 0, 1)  → Busy-wait (spin)            ║
║  Thread 2: atomicCAS(&lock, 0, 1)  → Gets lock                   ║
║            ...critical section...                                ║
║            atomicExch(&lock, 0)    → Release                     ║
║                                                                  ║
║  Features:                                                       ║
║  ✗ No OS scheduler (hardware scheduling only)                    ║
║  ✗ No context switching                                          ║
║  ✗ No fairness guarantees                                        ║
║  ✗ No deadlock detection                                         ║
║  ✗ No priorities                                                 ║
║  ✗ Threads spin (wastes GPU cycles)                              ║
║  ✓ Hardware-accelerated atomics                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Why Traditional Locks Don't Work Well on GPUs

1. **Massive Parallelism**: Thousands of threads competing for locks
2. **No Preemption**: A thread holding a lock can't be interrupted
3. **Warp Divergence**: Threads in same warp take different paths → inefficiency
4. **No Sleep Mechanism**: Threads must busy-wait, wasting cycles
5. **Scalability Issues**: Locks become bottlenecks with many threads

---

## Atomic Operations - The Preferred Approach {#atomic-operations}

### What Are Atomic Operations?

Atomic operations are **hardware-accelerated, indivisible** operations that complete without interruption. They are the **building blocks** of GPU synchronization.

### The Race Condition Problem

```
Without Atomics (WRONG):
────────────────────────

Time  Thread 0          Thread 1          Counter Value
────  ────────────      ────────────      ─────────────
  0   Read counter (5)                    5
  1                     Read counter (5)  5
  2   Add 1 → 6                           5
  3                     Add 1 → 6         5
  4   Write 6                             6  ← Lost update!
  5                     Write 6           6  ← Should be 7!
```

```
With Atomics (CORRECT):
───────────────────────

Time  Thread 0                    Thread 1                Counter
────  ──────────────────────      ──────────────────      ───────
  0   atomicAdd(&counter, 1)                              5
  1   → Read, add, write (6)                              6 ✓
  2   → Return old value (5)      atomicAdd(&counter, 1)  6
  3                               → Read, add, write (7)  7 ✓
  4                               → Return old value (6)  7 ✓
```

### Complete List of CUDA Atomic Operations

#### **Arithmetic Operations**

```cpp
// Addition and Subtraction
int old = atomicAdd(&address, value);     // *address += value
int old = atomicSub(&address, value);     // *address -= value

// Example: Counting events
__global__ void countEvents(int *counter, bool *events, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && events[idx]) {
        atomicAdd(counter, 1);  // Thread-safe increment
    }
}
```

#### **Comparison Operations**

```cpp
// Minimum and Maximum
int old = atomicMin(&address, value);     // *address = min(*address, value)
int old = atomicMax(&address, value);     // *address = max(*address, value)

// Example: Finding global maximum
__global__ void findMax(int *data, int *global_max, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicMax(global_max, data[idx]);  // Updates if data[idx] > *global_max
    }
}
```

#### **Increment/Decrement Operations**

```cpp
// Wraparound increment (useful for circular buffers)
unsigned old = atomicInc(&address, val);  
// If *address >= val: *address = 0
// Else: *address += 1

// Wraparound decrement
unsigned old = atomicDec(&address, val);
// If *address == 0 or *address > val: *address = val
// Else: *address -= 1
```

#### **Exchange Operations**

```cpp
// Simple exchange
int old = atomicExch(&address, value);    // old = *address; *address = value

// Compare-And-Swap (CAS) - Most powerful!
int old = atomicCAS(&address, compare, value);
// If *address == compare:
//     *address = value
// Return old value of *address

// Example: Lock implementation
__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {
        // Keep trying until we successfully change 0 → 1
    }
}
```

#### **Bitwise Operations**

```cpp
int old = atomicAnd(&address, value);     // *address &= value
int old = atomicOr(&address, value);      // *address |= value
int old = atomicXor(&address, value);     // *address ^= value

// Example: Setting bits in a bitmask
__global__ void setBits(unsigned int *mask, int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bit_position = indices[idx];
        atomicOr(mask, 1U << bit_position);  // Set bit atomically
    }
}
```

### Supported Data Types

```
╔═════════════════════════════════════════════════════════════════╗
║ Operation Type    │ int  │ unsigned │ long long │ float │ double║
╠═══════════════════╪══════╪══════════╪═══════════╪═══════╪═══════╣
║ Add/Sub           │  ✓   │    ✓     │     ✓     │   ✓   │   ✓   ║
║ Min/Max           │  ✓   │    ✓     │     ✓     │   ✗   │   ✗   ║
║ Inc/Dec           │  ✗   │    ✓     │     ✗     │   ✗   │   ✗   ║
║ Exch              │  ✓   │    ✓     │     ✓     │   ✓   │   ✗   ║
║ CAS               │  ✓   │    ✓     │     ✓     │   ✗   │   ✗   ║
║ And/Or/Xor        │  ✓   │    ✓     │     ✓     │   ✗   │   ✗   ║
╚═══════════════════╧══════╧══════════╧═══════════╧═══════╧═══════╝
```

### Performance Characteristics

```
Atomic Operation Speed (relative):
──────────────────────────────────

Shared Memory Atomics:    ████████████████████ 100% (baseline)
Global Memory Atomics:    ████████              40% (slower)
Regular Memory Access:    █████████████████████ 105% (fastest)

Key Insight: Atomics have overhead, but it's worth it for correctness!
```

### Real-World Example: Histogram Computation

#### **Naive Approach (WRONG)**

```cpp
__global__ void histogramNaive(unsigned char *image, int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned char pixel = image[idx];
        hist[pixel]++;  // ❌ RACE CONDITION!
    }
}
```

**Problem**: Multiple threads read-modify-write same bin → lost updates.

#### **Atomic Approach (CORRECT but can be slow)**

```cpp
__global__ void histogramAtomic(unsigned char *image, int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned char pixel = image[idx];
        atomicAdd(&hist[pixel], 1);  // ✓ Thread-safe
    }
}
```

**Issue**: High contention on global memory atomics (thousands of threads hitting same bins).

#### **Optimized Approach (BEST)**

```cpp
#define NUM_BINS 256

__global__ void histogramOptimized(unsigned char *image, int *hist, int n) {
    // Each block has its own histogram in shared memory
    __shared__ int localHist[NUM_BINS];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // Build histogram in shared memory (fast!)
    if (idx < n) {
        unsigned char pixel = image[idx];
        atomicAdd(&localHist[pixel], 1);  // Shared memory atomic (fast)
    }
    __syncthreads();
    
    // Merge into global histogram (only NUM_BINS atomics per block)
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (localHist[i] > 0) {
            atomicAdd(&hist[i], localHist[i]);  // Much fewer global atomics!
        }
    }
}
```

**Performance**:
```
Naive (wrong):        Fast but incorrect
Direct atomics:       ~15 ms (millions of global atomics)
Shared memory opt:    ~0.5 ms (only ~256 × num_blocks atomics)

Speedup: 30x faster! 🚀
```

---

## Spinlocks - When and Why to Avoid Them {#spinlocks}

### What is a Spinlock?

A spinlock is a **mutex-like synchronization primitive** built using atomic operations where threads **busy-wait** (spin) until the lock becomes available.

### Implementation

```cpp
struct Lock {
    int state;  // 0 = unlocked, 1 = locked
};

// Acquire lock
__device__ void lock(int *mutex) {
    // atomicCAS tries to change 0 → 1
    // If successful (returns 0), we got the lock
    // If unsuccessful (returns 1), someone else has it, keep trying
    while (atomicCAS(mutex, 0, 1) != 0) {
        // Spin (busy-wait)
        // Optionally: can add backoff here
    }
}

// Release lock
__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);  // Set to unlocked
}

// Usage
__global__ void criticalSectionExample(int *data, int *lock) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    lock(lock);
    
    // === Critical Section ===
    int temp = *data;
    temp = temp * 2 + 1;  // Some computation
    *data = temp;
    // ========================
    
    unlock(lock);
}
```

### The Warp Divergence Problem

```
╔═══════════════════════════════════════════════════════════════════╗
║                    THE WARP DIVERGENCE DISASTER                   ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  A warp has 32 threads that execute in lockstep (SIMT).           ║
║  With a lock, this is what happens:                               ║
║                                                                   ║
║  Warp 0 (32 threads):                                             ║
║  ┌────────────────────────────────────────────────────────────┐   ║
║  │ Thread 0: [Acquired lock] → Working in critical section    │   ║
║  │ Thread 1: [Spinning...] while(atomicCAS...) != 0           │   ║
║  │ Thread 2: [Spinning...] while(atomicCAS...) != 0           │   ║
║  │ Thread 3: [Spinning...] while(atomicCAS...) != 0           │   ║
║  │ ...                                                        │   ║
║  │ Thread 31: [Spinning...] while(atomicCAS...) != 0          │   ║
║  └────────────────────────────────────────────────────────────┘   ║
║                                                                   ║
║  Only 1/32 threads doing useful work = 3.125% efficiency! ❌      ║
║                                                                   ║
║  Wasted GPU Cycles: 96.875%                                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### Timeline Visualization

```
Time  Thread 0           Thread 1           Thread 2           Lock
────  ────────────       ────────────       ────────────       ────
  0   lock() → CAS       lock() → CAS       lock() → CAS       0
      Success! ✓         Failed             Failed             1
  
  1   [Critical          spin (CAS)         spin (CAS)         1
      Section]           Failed             Failed
  
  2   [Critical          spin (CAS)         spin (CAS)         1
      Section]           Failed             Failed
  
  3   [Critical          spin (CAS)         spin (CAS)         1
      Section]           Failed             Failed
  
  4   unlock()           spin (CAS)         spin (CAS)         0
                         Success! ✓         Failed             1
  
  5   [Finished]         [Critical          spin (CAS)         1
                         Section]           Failed
                         
Wasted Cycles: ███████████████████ (Threads 1-31 spinning)
Useful Work:   ██ (Only Thread 0 working)
```

### Why Spinlocks Are Terrible on GPUs

#### **1. Warp Divergence**

```
Without Lock (All threads work):
████████████████████████████████ 100% efficiency

With Lock (Only 1 works):
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3.125% efficiency

Performance Loss: 96.875%! 💥
```

#### **2. Deadlock Risk**

```
Scenario: Thread 0 acquires lock, then GPU schedules different warp
         → Thread 0 can't finish critical section
         → Other threads spin forever
         → DEADLOCK! 💀

GPU has NO OS scheduler to detect or break deadlocks!
```

#### **3. Scalability Disaster**

```
Performance vs Number of Threads:

Threads      Time (ms)    Throughput
──────────   ─────────    ──────────
   1,000        10           100 ops/ms
  10,000       100            10 ops/ms  ← 10x worse
 100,000     1,000             1 ops/ms  ← 100x worse!

Locks DON'T SCALE on GPUs!
```

#### **4. Priority Inversion**

```
High Priority Thread: Waiting for lock held by low priority thread
Low Priority Thread:  Scheduled out by GPU, can't release lock
Result:              High priority thread stuck! No solution!
```

### When Spinlocks Might Be Acceptable

Use spinlocks ONLY if **ALL** these conditions are true:

```
✓ Critical section is EXTREMELY short (< 10 instructions)
✓ Very low contention (< 1% of threads need lock simultaneously)
✓ No alternative (atomics can't express the operation)
✓ You've profiled and confirmed it's not a bottleneck
✓ Willing to accept warp divergence

Even then, consider redesigning to avoid locks!
```

### Better Alternatives to Spinlocks

```cpp
// Instead of:
lock(&mutex);
counter++;
unlock(&mutex);

// Use:
atomicAdd(&counter, 1);  // 100x faster!

// Instead of:
lock(&mutex);
max_val = max(max_val, new_val);
unlock(&mutex);

// Use:
atomicMax(&max_val, new_val);  // Much faster!

// Instead of:
lock(&mutex);
if (condition) {
    complex_update();
}
unlock(&mutex);

// Redesign algorithm to be lock-free!
// Use atomicCAS in a loop with read-compute-CAS pattern
```

---

## Semaphores in GPU Programming {#semaphores}

### What is a Semaphore?

A **semaphore** is a synchronization primitive that controls access to a resource pool with **N available slots**.

```
Binary Semaphore (N=1):    Just like a mutex/lock
Counting Semaphore (N>1):  Multiple threads can enter
```

### Visual Example: Counting Semaphore (N=3)

```
╔════════════════════════════════════════════════════════════════╗
║           SEMAPHORE WITH 3 AVAILABLE SLOTS                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Time 0: Available = [3]  ░░░ (3 free slots)                   ║
║  ──────────────────────────────────────────────────────────    ║
║  Thread 0 enters:  [2]  ░░▓ (Thread 0 using slot)              ║
║  Thread 1 enters:  [1]  ░▓▓ (Threads 0,1 using slots)          ║
║  Thread 2 enters:  [0]  ▓▓▓ (All slots full!)                  ║
║  Thread 3 waits... [0]  ▓▓▓ (Must wait for a slot)             ║
║                                                                ║
║  Thread 0 exits:   [1]  ░▓▓ (Slot freed)                       ║
║  Thread 3 enters:  [0]  ▓▓▓ (Thread 3 takes freed slot)        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Implementation

```cpp
// Initialize semaphore
__device__ void semaphore_init(int *sem, int count) {
    *sem = count;
}

// Wait (P operation, acquire)
__device__ void semaphore_wait(int *sem) {
    int old;
    do {
        old = *sem;
        
        // If no slots available, keep trying
        if (old <= 0) {
            continue;
        }
        
        // Try to decrement (acquire a slot)
        // If successful, atomicCAS returns old value
    } while (atomicCAS(sem, old, old - 1) != old);
}

// Signal (V operation, release)
__device__ void semaphore_signal(int *sem) {
    atomicAdd(sem, 1);  // Release a slot
}

// Usage example: Limit concurrent access
__global__ void limitedAccessKernel(int *sem, int *shared_resource) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only N threads can enter this section at a time
    semaphore_wait(sem);
    
    // === Limited Concurrent Section ===
    // Do work with shared_resource
    // Max N threads here simultaneously
    // ===================================
    
    semaphore_signal(sem);  // Release slot for others
}
```

### Use Case: Rate Limiting

```cpp
// Limit to 100 concurrent memory allocations
__shared__ int allocation_sem;

if (threadIdx.x == 0) {
    allocation_sem = 100;  // 100 slots
}
__syncthreads();

// Each thread wants to allocate
semaphore_wait(&allocation_sem);

// Allocate (only 100 threads do this simultaneously)
void *ptr = allocate_memory();

// Use allocation...

semaphore_signal(&allocation_sem);
```

### Semaphore vs Lock

```
╔══════════════════════════════════════════════════════════════╗
║ Feature          │ Lock (N=1)      │ Semaphore (N>1)         ║
╠══════════════════╪═════════════════╪═════════════════════════╣
║ Concurrent       │ 1 thread        │ N threads               ║
║ Access           │                 │                         ║
║ ─────────────────┼─────────────────┼─────────────────────────║
║ Use Case         │ Mutual          │ Resource pool           ║
║                  │ exclusion       │ management              ║
║ ─────────────────┼─────────────────┼─────────────────────────║
║ Typical N        │ 1               │ 10-1000                 ║
║ ─────────────────┼─────────────────┼─────────────────────────║
║ GPU Performance  │ ❌ Poor         │ ⚠️ Better but still     ║
║                  │                 │    has overhead         ║
╚══════════════════════════════════════════════════════════════╝
```

### Warning: Same Issues as Spinlocks

Semaphores on GPU suffer from **similar problems** as spinlocks:
- ❌ Busy-waiting wastes cycles
- ❌ Warp divergence when waiting
- ❌ No deadlock detection
- ❌ Poor scalability

**Recommendation**: Use lock-free algorithms instead!

---

## Lock-Free Algorithms {#lock-free-algorithms}

### What Makes an Algorithm Lock-Free?

A lock-free algorithm guarantees **system-wide progress** even if individual threads are delayed. Uses atomics but no locks.

```
With Locks:                    Lock-Free:
────────────                   ──────────

Thread 1: Acquired lock       Thread 1: atomicCAS loop
Thread 2: Waiting...          Thread 2: atomicCAS loop
Thread 3: Waiting...          Thread 3: atomicCAS loop
Thread 4: Waiting...          Thread 4: atomicCAS loop

If Thread 1 stalls:           If Thread 1 stalls:
→ ALL threads blocked! ❌      → Others keep trying! ✓
→ System deadlock             → System makes progress
```

### Key Properties

```
✓ No locks or mutexes
✓ Uses atomic compare-and-swap (CAS)
✓ Guaranteed progress for at least one thread
✓ No deadlocks possible
✓ Better scalability
✓ Warp-friendly (less divergence)
```

### Pattern: Read-Modify-Write with CAS

```cpp
// Generic lock-free update pattern
__device__ void lockFreeUpdate(int *address, 
                               int (*modify_function)(int)) {
    int old, new_val;
    
    do {
        old = *address;                    // Read current value
        new_val = modify_function(old);    // Compute new value
        
        // Try to update if value hasn't changed
        // If successful, atomicCAS returns old
        // If failed (someone else updated), retry
    } while (atomicCAS(address, old, new_val) != old);
}
```

### Example 1: Lock-Free Maximum

```cpp
// Find maximum value in array
__global__ void lockFreeMax(int *data, int *global_max, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int my_value = data[idx];
        
        // atomicMax is lock-free!
        atomicMax(global_max, my_value);
        
        // Equivalent manual implementation:
        /*
        int old, new_val;
        do {
            old = *global_max;
            new_val = max(old, my_value);
        } while (atomicCAS(global_max, old, new_val) != old);
        */
    }
}
```

### Example 2: Lock-Free Counter

```cpp
__global__ void lockFreeCount(int *counter, bool *conditions, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && conditions[idx]) {
        // atomicAdd is lock-free!
        atomicAdd(counter, 1);
        
        // What's happening internally:
        // 1. Read current value
        // 2. Add 1
        // 3. Try to write back
        // 4. If someone else updated, retry from step 1
    }
}
```

### Example 3: Lock-Free Stack (Advanced)

```cpp
struct Node {
    int data;
    Node *next;
};

// Lock-free stack push
__device__ void lockFreePush(Node **top, Node *new_node) {
    Node *old_top;
    
    do {
        old_top = *top;              // Read current top
        new_node->next = old_top;    // Point new node to current top
        
        // Try to make new_node the new top
        // If *top is still old_top, update to new_node
        // If not, someone else pushed, retry
    } while (atomicCAS((unsigned long long*)top, 
                       (unsigned long long)old_top,
                       (unsigned long long)new_node) != 
             (unsigned long long)old_top);
}

// Lock-free stack pop
__device__ Node* lockFreePop(Node **top) {
    Node *old_top, *new_top;
    
    do {
        old_top = *top;              // Read current top
        
        if (old_top == NULL) {
            return NULL;             // Stack empty
        }
        
        new_top = old_top->next;     // Next node becomes new top
        
        // Try to update top
    } while (atomicCAS((unsigned long long*)top,
                       (unsigned long long)old_top,
                       (unsigned long long)new_top) !=
             (unsigned long long)old_top);
    
    return old_top;
}
```

### Example 4: Lock-Free Linked List Insert

```cpp
__device__ void lockFreeInsert(Node **head, Node *new_node, int key) {
    Node *curr, *next;
    
    while (true) {
        // Find insertion point
        curr = *head;
        
        while (curr != NULL && curr->data < key) {
            curr = curr->next;
        }
        
        // Try to insert
        new_node->next = curr;
        
        if (atomicCAS((unsigned long long*)head,
                      (unsigned long long)curr,
                      (unsigned long long)new_node) ==
            (unsigned long long)curr) {
            break;  // Success!
        }
        
        // Failed, retry (someone else modified list)
    }
}
```

### Performance Comparison

```
Operation: Increment counter 1,000,000 times

╔════════════════════════════════════════════════════════════╗
║ Method           │ Time    │ Throughput  │ Correctness     ║
╠══════════════════╪═════════╪═════════════╪═════════════════╣
║ No sync (naive)  │ 0.5 ms  │ 2000 M/s    │ ❌ WRONG        ║
║ Spinlock         │ 850 ms  │ 1.2 M/s     │ ✓ Correct       ║
║ Lock-free atomic │ 4.2 ms  │ 238 M/s     │ ✓ Correct       ║
╚════════════════════════════════════════════════════════════╝

Lock-free is 200x faster than spinlock! 🚀
```

### Advantages of Lock-Free Algorithms

```
✓ No deadlocks ever
✓ No priority inversion
✓ Better scalability
✓ Reduced warp divergence
✓ Composable (can combine multiple operations)
✓ Progress guarantee
✓ Lower latency
✓ Hardware-accelerated atomics
```

### When to Use Lock-Free

```
Use lock-free when:
───────────────────
✓ Need to update shared counters
✓ Finding max/min across threads
✓ Building histograms
✓ Accumulating results
✓ Managing work queues
✓ Implementing data structures

Avoid locks when:
────────────────
✗ GPU is involved (almost always use lock-free instead!)
```

---

## Warp-Level Synchronization {#warp-level-synchronization}

### Understanding Warps

A **warp** is a group of **32 threads** that execute together in **lockstep** (SIMT - Single Instruction, Multiple Threads).

```
╔══════════════════════════════════════════════════════════════╗
║                        WARP EXECUTION                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Block with 256 threads = 8 warps                            ║
║                                                              ║
║  Warp 0:  Threads  0-31   ████████████████████████████████   ║
║  Warp 1:  Threads 32-63   ████████████████████████████████   ║
║  Warp 2:  Threads 64-95   ████████████████████████████████   ║
║  ...                                                         ║
║  Warp 7:  Threads 224-255 ████████████████████████████████   ║
║                                                              ║
║  All threads in a warp execute the SAME instruction!         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Warp-Level Primitives

#### **1. Warp Synchronization**

```cpp
// Synchronize all threads in warp
__syncwarp();              // All threads in warp (0xFFFFFFFF)
__syncwarp(0x0000FFFF);    // Only lower 16 threads

// Not needed for warp shuffles (they're implicitly synchronous)
```

#### **2. Warp Shuffle - Data Exchange**

```cpp
// Shuffle data between lanes in a warp
__shfl_sync(mask, var, srcLane);      // Get var from srcLane
__shfl_up_sync(mask, var, delta);     // Get var from (lane - delta)
__shfl_down_sync(mask, var, delta);   // Get var from (lane + delta)
__shfl_xor_sync(mask, var, laneMask); // Get var from (lane ^ laneMask)

// mask: Which threads participate (usually 0xFFFFFFFF = all 32)
```

**Visual Example: `__shfl_down_sync`**

```
Initial State (each lane has its ID):
Lane:  0   1   2   3   4   5   6   7  ... 31
Value: 0   1   2   3   4   5   6   7  ... 31

After: value = __shfl_down_sync(0xFFFFFFFF, value, 1):
Lane:  0   1   2   3   4   5   6   7  ... 31
Value: 1   2   3   4   5   6   7   8  ... 31  (each got value from lane+1)

After: value = __shfl_down_sync(0xFFFFFFFF, value, 2):
Lane:  0   1   2   3   4   5   6   7  ... 31
Value: 2   3   4   5   6   7   8   9  ... 31  (each got value from lane+2)
```

#### **3. Warp Vote Functions**

```cpp
// Voting across warp
int all_true  = __all_sync(mask, predicate);    // All threads true?
int any_true  = __any_sync(mask, predicate);    // Any thread true?
unsigned mask = __ballot_sync(mask, predicate); // Bitmask of true threads

// Example: Check if all threads found valid data
bool found = (data[idx] > 0);
if (__all_sync(0xFFFFFFFF, found)) {
    // All 32 threads in warp found valid data!
}
```

**Visual Example: `__ballot_sync`**

```
Predicate per thread:
Lane:  0   1   2   3   4   5   6   7  ...
Pred:  T   F   T   T   F   F   T   F  ...

Result = __ballot_sync(0xFFFFFFFF, predicate):
Binary:  1   0   1   1   0   0   1   0  ...
Result = 0b10110010... (as 32-bit integer)

Each bit represents one thread's vote!
```

### Warp-Level Reduction (NO LOCKS!)

```cpp
// Sum reduction across warp - extremely fast!
__device__ int warpReduceSum(int val) {
    // All threads in warp participate
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // Lane 0 has final sum
}

// Visual execution:
// Initial: [1, 2, 3, 4, 5, 6, 7, 8, ...]
//
// Step 1 (offset=16):
//   Lane 0-15 get values from Lane 16-31
//   [1+17, 2+18, 3+19, ..., 16+32, 17, 18, ...]
//
// Step 2 (offset=8):
//   Lane 0-7 get values from Lane 8-15
//   ...
//
// Step 5 (offset=1):
//   Lane 0 gets value from Lane 1
//   Lane 0 now has sum of all 32 values!
```

### Complete Example: Parallel Reduction

```cpp
__global__ void warpLevelSum(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;      // Lane within warp (0-31)
    int warp_id = threadIdx.x / 32;   // Which warp in block
    
    // Each thread loads one value
    int val = (idx < n) ? input[idx] : 0;
    
    // Warp-level reduction (no atomics, no shared memory!)
    val = warpReduceSum(val);
    
    // Only first thread in each warp has the sum
    if (lane == 0) {
        // Could use shared memory here to combine warps in block
        // Or use atomic for final global sum
        atomicAdd(output, val);
    }
}
```

### Warp-Level Max/Min

```cpp
__device__ int warpReduceMax(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = max(val, other);
    }
    return val;  // Lane 0 has maximum
}

__device__ int warpReduceMin(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = min(val, other);
    }
    return val;  // Lane 0 has minimum
}
```

### Performance Benefits

```
Traditional Reduction (shared memory + atomics):
──────────────────────────────────────────────
1. Load to shared memory    → Memory access
2. __syncthreads()          → Barrier
3. Reduce in shared memory  → Memory access
4. __syncthreads()          → Barrier
5. Atomic to global         → Atomic contention

Warp Shuffle Reduction:
───────────────────────
1. Load to register         → Register (fast!)
2. Shuffle in registers     → Register (fast!)
3. Atomic to global         → Atomic contention

Speedup: 2-5x faster! 🚀
No shared memory needed → More cache available
```

### Key Insights

```
✓ Warps execute in lockstep → No synchronization needed!
✓ Shuffle instructions are extremely fast (register-to-register)
✓ No shared memory → Better cache utilization
✓ No __syncthreads() → Lower latency
✓ Perfect for small reductions (32 elements)
✓ Combine with block-level for larger reductions
```

---

## Block-Level Synchronization {#block-level-synchronization}

### `__syncthreads()` - The Block Barrier

The `__syncthreads()` function creates a **barrier** where all threads in a block must arrive before any can proceed.

```
╔════════════════════════════════════════════════════════════════╗
║                    __syncthreads() BARRIER                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Thread 0    Thread 1    Thread 2    Thread 3   ...  Thread N  ║
║     ║            ║            ║            ║            ║      ║
║     ║ Write      ║ Write      ║ Write      ║ Write      ║      ║
║     ║            ║            ║            ║            ║      ║
║     ▼            ▼            ▼            ▼            ▼      ║
║  ╔══════════════════════════════════════════════════════════╗  ║
║  ║              __syncthreads() BARRIER                     ║  ║
║  ║  All threads MUST reach here before ANY can continue     ║  ║
║  ╚══════════════════════════════════════════════════════════╝  ║
║     ▼            ▼            ▼            ▼            ▼      ║
║     ║ Read       ║ Read       ║ Read       ║ Read       ║      ║
║     ║            ║            ║            ║            ║      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Use Case: Shared Memory Coordination

```cpp
__global__ void sharedMemoryExample(float *input, float *output, int n) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: All threads write to shared memory
    if (idx < n) {
        shared[tid] = input[idx];
    }
    
    // BARRIER: Wait for all writes to complete
    __syncthreads();
    // After this point, all shared[] elements are valid
    
    // Phase 2: All threads read from shared memory
    if (idx < n) {
        // Can safely read any element
        float left  = (tid > 0) ? shared[tid - 1] : 0.0f;
        float center = shared[tid];
        float right = (tid < blockDim.x - 1) ? shared[tid + 1] : 0.0f;
        
        output[idx] = (left + center + right) / 3.0f;
    }
}
```

### Common Pattern: Multi-Stage Processing

```cpp
__global__ void multiStageKernel(int *data, int n) {
    __shared__ int temp[256];
    
    int tid = threadIdx.x;
    
    // Stage 1: Load and transform
    temp[tid] = data[tid] * 2;
    __syncthreads();  // ← Wait for stage 1
    
    // Stage 2: Aggregate neighbors
    int sum = temp[tid];
    if (tid > 0) sum += temp[tid - 1];
    if (tid < 255) sum += temp[tid + 1];
    temp[tid] = sum;
    __syncthreads();  // ← Wait for stage 2
    
    // Stage 3: Write back
    data[tid] = temp[tid];
}
```

### Parallel Reduction with `__syncthreads()`

```cpp
__global__ void blockReduceSum(int *input, int *output, int n) {
    __shared__ int shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();  // ← Wait for all loads
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();  // ← Wait after each reduction step
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Visual execution (8 threads):
// Initial:  [5, 3, 7, 2, 9, 1, 4, 6]
//            ↓ stride=4
// Step 1:   [14, 4, 11, 8, ...] (5+9, 3+1, 7+4, 2+6)
//            __syncthreads()
//            ↓ stride=2
// Step 2:   [25, 12, ...] (14+11, 4+8)
//            __syncthreads()
//            ↓ stride=1
// Step 3:   [37, ...] (25+12)
//            __syncthreads()
// Result:   shared[0] = 37 ✓
```

### Critical Rules for `__syncthreads()`

#### **Rule 1: ALL threads must reach barrier**

```cpp
// ❌ WRONG: Conditional barrier
if (threadIdx.x < 128) {
    __syncthreads();  // Only some threads reach here → DEADLOCK!
}

// ✓ CORRECT: Unconditional barrier
__syncthreads();  // All threads reach here
if (threadIdx.x < 128) {
    // Do work after barrier
}
```

#### **Rule 2: Doesn't sync across blocks**

```cpp
// ❌ WRONG: Trying to sync different blocks
__global__ void wrongSync(int *data) {
    int block_id = blockIdx.x;
    
    // Do work in block 0
    if (block_id == 0) {
        data[0] = 1;
    }
    
    __syncthreads();  // Only syncs threads within SAME block!
    
    // Block 1 might see old value!
    if (block_id == 1) {
        int val = data[0];  // Race condition! Might be 0 or 1
    }
}

// ✓ CORRECT: Use separate kernel launches
// Kernel launches provide implicit global synchronization
kernel1<<<...>>>(data);  // All blocks finish
cudaDeviceSynchronize(); // Explicit host sync
kernel2<<<...>>>(data);  // All blocks start fresh
```

#### **Rule 3: Beware of warp divergence**

```cpp
// ⚠️ TRICKY: Different warps might behave differently
__global__ void trickySync(int *data) {
    __shared__ int shared[256];
    
    int tid = threadIdx.x;
    
    shared[tid] = data[tid];
    
    // This is OK even though it looks conditional
    // because all threads in block execute __syncthreads()
    if (tid % 2 == 0) {
        __syncthreads();
        // Do something
    } else {
        __syncthreads();
        // Do something else
    }
    // Both branches have __syncthreads() at same depth → OK
}
```

### Performance Characteristics

```
__syncthreads() Cost:
─────────────────────
Latency:     ~20-100 cycles
Overhead:    Increases with block size
Alternative: Warp-level ops (if possible)

Block Size   Sync Cost
──────────   ─────────
32 threads   ~20 cycles
128 threads  ~40 cycles
256 threads  ~80 cycles
512 threads  ~150 cycles

Recommendation: Use __syncthreads() when needed,
                but minimize frequency
```

### Advanced: Counting Semaphore with `__syncthreads()`

```cpp
__global__ void blockSemaphore(int *data) {
    __shared__ int semaphore;
    __shared__ int queue[256];
    
    int tid = threadIdx.x;
    
    // Initialize
    if (tid == 0) {
        semaphore = 10;  // 10 concurrent slots
    }
    __syncthreads();
    
    // Each thread tries to acquire
    bool acquired = false;
    while (!acquired) {
        int old = atomicAdd(&semaphore, -1);
        if (old > 0) {
            acquired = true;
        } else {
            atomicAdd(&semaphore, 1);  // Put it back
            __syncthreads();  // Wait before retry
        }
    }
    
    // Critical section (max 10 threads here)
    // ...
    
    // Release
    atomicAdd(&semaphore, 1);
}
```

---

## Grid-Level Synchronization {#grid-level-synchronization}

### The Challenge

**Problem**: No built-in mechanism to synchronize ALL blocks in a grid!

```
╔════════════════════════════════════════════════════════════════╗
║           Block 0        Block 1        Block 2      Block N   ║
║           ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐   ║
║           │Thread│      │Thread│      │Thread│      │Thread│   ║
║           │ ...  │      │ ...  │      │ ...  │      │ ...  │   ║
║           └──────┘      └──────┘      └──────┘      └──────┘   ║
║               ║             ║             ║             ║      ║
║               ▼             ▼             ▼             ▼      ║
║          __syncthreads() works within each block               ║
║                                                                ║
║          ❌ NO way to sync across blocks! ❌                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Solution 1: Multiple Kernel Launches (Recommended)

```cpp
// Kernel launches provide implicit global synchronization
__global__ void phase1(int *data, int n) {
    // All blocks work on data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
}

__global__ void phase2(int *data, int n) {
    // All blocks see results from phase1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1;
    }
}

// Host code
phase1<<<gridSize, blockSize>>>(d_data, n);
cudaDeviceSynchronize();  // Wait for ALL blocks to finish
phase2<<<gridSize, blockSize>>>(d_data, n);
cudaDeviceSynchronize();

// Phase1 → [ALL blocks finish] → Phase2
// Perfect synchronization! ✓
```

**Advantages:**
- ✓ Guaranteed synchronization
- ✓ Clean and simple
- ✓ Matches GPU architecture
- ✓ No risk of deadlock

**Disadvantages:**
- ⚠️ Kernel launch overhead (~10-20 μs)
- ⚠️ State must be in global memory

### Solution 2: Cooperative Groups (Modern CUDA)

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(int *data, int n) {
    // Get grid group (all threads in grid)
    cg::grid_group grid = cg::this_grid();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
    
    // Synchronize ALL threads in grid!
    grid.sync();  // ← Grid-wide barrier
    
    // Phase 2
    if (idx < n) {
        data[idx] = data[idx] + 1;
    }
}

// Must launch with cooperative groups API
void *args[] = {&d_data, &n};
cudaLaunchCooperativeKernel((void*)cooperativeKernel,
                           gridSize, blockSize,
                           args);
```

**Requirements:**
- Compute capability >= 6.0
- Special launch API
- Grid size limits (must fit on GPU simultaneously)

### Solution 3: Atomic Counter Barrier (Risky!)

```cpp
__device__ void gridBarrier(int *counter, int num_blocks) {
    __shared__ bool am_i_last;
    
    if (threadIdx.x == 0) {
        // Atomic increment counter
        int old = atomicAdd(counter, 1);
        
        // Am I the last block?
        am_i_last = (old == num_blocks - 1);
        
        if (am_i_last) {
            // Reset for next barrier
            *counter = 0;
        }
    }
    __syncthreads();  // Wait for thread 0
    
    // Last block proceeds, others spin
    while (!am_i_last) {
        // Busy-wait
        am_i_last = (*counter == 0);  // Reset by last block
    }
}

__global__ void kernelWithGridBarrier(int *data, int *counter, int n) {
    int num_blocks = gridDim.x * gridDim.y * gridDim.z;
    
    // Phase 1
    // ...
    
    gridBarrier(counter, num_blocks);  // ← Grid-wide sync
    
    // Phase 2
    // ...
}
```

**⚠️ WARNING**: This is DANGEROUS!
- Can deadlock if blocks don't fit on GPU simultaneously
- Wastes cycles (busy-waiting)
- No forward progress guarantee
- **DON'T USE IN PRODUCTION**

### Solution 4: Persistent Kernels

```cpp
__global__ void persistentKernel(int *data, int *work_queue, 
                                  int num_tasks) {
    while (true) {
        // Atomically get next task
        int task_id = atomicAdd(work_queue, 1);
        
        if (task_id >= num_tasks) {
            break;  // No more work
        }
        
        // Process task
        // ...
    }
    
    // Implicit grid-wide synchronization at kernel end
}
```

**Benefits:**
- No repeated kernel launches
- Amortizes launch overhead
- Dynamic load balancing

### Comparison

```
╔══════════════════════════════════════════════════════════════╗
║ Method              │ Safety  │ Performance │ Complexity     ║
╠═════════════════════╪═════════╪═════════════╪════════════════╣
║ Multiple Kernels    │ ✓✓✓     │ Good        │ Simple         ║
║ Cooperative Groups  │ ✓✓✓     │ Better      │ Moderate       ║
║ Atomic Barrier      │ ❌ BAD  │ Poor        │ Simple         ║
║ Persistent Kernel   │ ✓✓      │ Best        │ Complex        ║
╚══════════════════════════════════════════════════════════════╝

Recommendation: Use multiple kernel launches (Solution 1)
```

### When Do You Need Grid-Level Sync?

```
Common Use Cases:
─────────────────
✓ Multi-phase algorithms (e.g., parallel prefix scan)
✓ Iterative algorithms (e.g., Jacobi iteration)
✓ Graph algorithms (BFS levels)
✓ Sorting networks
✓ Reduce-then-broadcast patterns

How to Handle:
──────────────
1. Split into multiple kernels (best)
2. Use cooperative groups (modern)
3. Redesign algorithm to avoid global sync
4. Use persistent kernels with work stealing
```

---

## Performance Comparison {#performance-comparison}

### Benchmark Setup

```
Test: Increment counter 1,000,000 times
Hardware: NVIDIA RTX 3090
Block Size: 256 threads
Grid Size: 4096 blocks
```

### Results

```
╔══════════════════════════════════════════════════════════════════════╗
║ Method              │ Time     │ Throughput  │ Correct? │ Efficiency ║
╠═════════════════════╪══════════╪═════════════╪══════════╪════════════╣
║ Naive (no sync)     │   0.5 ms │ 2000 M/s    │ ❌ NO    │ N/A        ║
║ ────────────────────┼──────────┼─────────────┼──────────┼────────────║
║ Global Spinlock     │ 850  ms  │  1.2 M/s    │ ✓ YES    │  0.06%     ║
║ Block Spinlocks     │ 220  ms  │  4.5 M/s    │ ✓ YES    │  0.23%     ║
║ ────────────────────┼──────────┼─────────────┼──────────┼────────────║
║ Global Atomic       │   4.2 ms │  238 M/s    │ ✓ YES    │ 11.9%      ║
║ Shared→Global Atomic│   0.6 ms │ 1667 M/s    │ ✓ YES    │ 83.4%      ║
║ ────────────────────┼──────────┼─────────────┼──────────┼────────────║
║ Warp Shuffle        │   0.3 ms │ 3333 M/s    │ ✓ YES    │ 166.7%     ║
║ (with atomic final) │          │             │          │            ║
╚══════════════════════════════════════════════════════════════════════╝

Key Insights:
─────────────
• Spinlock: 2000x slower than naive! 😱
• Optimized atomic: 14x faster than global atomic
• Warp shuffle: Fastest correct implementation
• Shared memory staging: Critical for atomics performance
```

### Detailed Analysis

#### **Why Spinlock Is So Slow**

```
1,000,000 increments with spinlock:

Threads competing: 1,048,576 (4096 blocks × 256 threads)
Threads holding lock at once: 1

Average wait time per thread:
= (Total threads / Throughput) 
= 1,048,576 / 1,200,000
= 0.87 seconds per thread waiting!

Wasted GPU cycles:
= (1,048,576 - 1) / 1,048,576
= 99.9999% of threads just spinning! 💥
```

#### **Why Warp Shuffle Wins**

```
Warp Shuffle Breakdown:
───────────────────────

1. Each warp (32 threads) reduces locally
   → 32 values → 1 value
   → Uses registers only (fast!)
   → Takes ~10 cycles
   
2. Each block (8 warps) has 8 partial sums
   → 8 atomics to global memory
   → Much less contention!
   
3. Total atomics: 4096 blocks × 8 = 32,768 atomics
   vs. 1,000,000 atomics for direct approach
   
Atomic reduction: 30x fewer atomics! 🚀
```

### Memory Hierarchy Impact

```
Operation Location         Latency    Bandwidth    Contention
───────────────────────    ────────   ──────────   ──────────
Register (warp shuffle)    1 cycle    ~20 TB/s     None
Shared memory atomic       ~30 cycles ~1.5 TB/s    Low
Global memory atomic       ~400 cycles ~900 GB/s   High
Spinlock                   Variable   N/A          Extreme

Best → Worst:
1. Register ops (warp shuffle)
2. Shared memory atomics  
3. Global memory atomics
4. Locks (avoid!)
```

### Scalability Analysis

```
Performance vs. Number of Threads:

Threads     Spinlock    Atomic    Warp Shuffle
──────────  ──────────  ────────  ────────────
1,024       5 ms        2 ms      0.1 ms
10,240      50 ms       3 ms      0.2 ms
102,400     500 ms      4 ms      0.3 ms
1,024,000   5000 ms     5 ms      0.4 ms

Scalability:
• Spinlock: O(n²) - gets MUCH worse!
• Atomic: O(n) - linear degradation
• Warp shuffle: O(log n) - barely increases!
```

---

## Best Practices and Guidelines {#best-practices}

### The Golden Rule

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                   PREFER LOCK-FREE!                          ║
║                                                              ║
║  If you find yourself reaching for a lock on GPU,            ║
║  stop and redesign your algorithm to be lock-free.           ║
║                                                              ║
║  99% of the time, there's a better way.                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Decision Tree

```
Need to synchronize?
    │
    ├─ Within warp (32 threads)?
    │  └─ Use warp shuffles (__shfl_sync) ✓
    │
    ├─ Within block (up to 1024 threads)?
    │  └─ Use __syncthreads() ✓
    │
    ├─ Across blocks but simple operation?
    │  ├─ Counter → atomicAdd ✓
    │  ├─ Max/Min → atomicMax/Min ✓
    │  ├─ Histogram → atomicAdd per bin ✓
    │  └─ Complex → Try to decompose into atomics
    │
    ├─ Across blocks, complex operation?
    │  ├─ Multiple kernel launches ✓
    │  ├─ Cooperative groups (if supported) ✓
    │  └─ Redesign algorithm? ✓
    │
    └─ Really need a lock?
       └─ Are you ABSOLUTELY sure?
          └─ Okay, but expect poor performance ⚠️
```

### Optimization Hierarchy

```
Level 1: Algorithm Design (MOST IMPORTANT)
─────────────────────────────────────────
✓ Design lock-free algorithms from the start
✓ Use embarrassingly parallel patterns when possible
✓ Minimize shared state
✓ Partition data to avoid conflicts

Level 2: Synchronization Primitive Selection
─────────────────────────────────────────────
✓ Warp-level ops > Block-level > Grid-level
✓ Atomics > Locks
✓ Shared memory > Global memory

Level 3: Contention Reduction
──────────────────────────────
✓ Use shared memory to stage atomics
✓ Privatize per-block/warp copies
✓ Reduce atomic operation frequency

Level 4: Low-Level Optimization
────────────────────────────────
✓ Minimize atomic scope (shared vs global)
✓ Use appropriate data types
✓ Consider backoff strategies
```

### Checklist Before Using Locks

```
□ Have you tried atomicAdd/Sub?
□ Have you tried atomicMax/Min?
□ Have you tried atomicCAS in a loop?
□ Can you use warp shuffles instead?
□ Can you stage through shared memory?
□ Can you split into multiple kernels?
□ Can you use cooperative groups?
□ Have you profiled to confirm locks aren't the bottleneck?
□ Have you considered algorithm redesign?
□ Is the critical section EXTREMELY short (<10 instructions)?
□ Is contention very low (<1% of threads)?

If you answered NO to any of these, DON'T USE LOCKS!
```

### Common Mistakes to Avoid

#### **Mistake 1: Using locks for simple operations**

```cpp
// ❌ BAD
lock(&mutex);
counter++;
unlock(&mutex);

// ✓ GOOD
atomicAdd(&counter, 1);
```

#### **Mistake 2: High-contention locks**

```cpp
// ❌ BAD: All threads fight for one lock
__global__ void bad(int *lock, int *data) {
    lock(lock);
    *data += 1;
    unlock(lock);
}

// ✓ GOOD: Use atomics
__global__ void good(int *data) {
    atomicAdd(data, 1);
}
```

#### **Mistake 3: Conditional `__syncthreads()`**

```cpp
// ❌ BAD: Deadlock!
if (threadIdx.x < 64) {
    __syncthreads();  // Only some threads reach
}

// ✓ GOOD
__syncthreads();
if (threadIdx.x < 64) {
    // Work
}
```

#### **Mistake 4: Trying to sync across blocks**

```cpp
// ❌ BAD
__syncthreads();  // Only syncs within block!

// ✓ GOOD
// Use multiple kernel launches
```

#### **Mistake 5: Ignoring warp divergence**

```cpp
// ❌ BAD: Half warp spins
if (threadIdx.x % 2 == 0) {
    lock(&mutex);
    // Critical section
    unlock(&mutex);
}

// ✓ GOOD: Redesign to avoid locks entirely
```

### Performance Tips

```
1. Reduce Atomic Contention
   ─────────────────────────
   • Stage through shared memory
   • Use per-block atomics, then reduce
   • Coarsen granularity (batch operations)

2. Optimize Critical Sections
   ──────────────────────────
   • Keep them SHORT (<10 instructions)
   • Move non-critical work outside
   • Use read-modify-write atomics

3. Memory Location Matters
   ───────────────────────
   Shared memory atomic: 30 cycles
   Global memory atomic: 400 cycles
   → Use shared memory when possible!

4. Warp-Level First
   ────────────────
   • Reduce within warp using shuffles
   • Only use atomics for inter-warp/block
   • Minimizes atomic operations

5. Profile and Measure
   ───────────────────
   • Use Nsight Compute
   • Check for atomic bottlenecks
   • Measure actual impact
```

---

## Common Patterns and Use Cases {#common-patterns}

### Pattern 1: Global Counter

```cpp
// Initialize
int *d_counter;
cudaMalloc(&d_counter, sizeof(int));
cudaMemset(d_counter, 0, sizeof(int));

// Kernel
__global__ void countEvents(bool *conditions, int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && conditions[idx]) {
        atomicAdd(counter, 1);
    }
}

// Better: Warp-level reduction first
__global__ void countEventsOptimized(bool *conditions, int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    // Each thread has 0 or 1
    int my_count = (idx < n && conditions[idx]) ? 1 : 0;
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_count += __shfl_down_sync(0xFFFFFFFF, my_count, offset);
    }
    
    // Only lane 0 atomics
    if (lane == 0 && my_count > 0) {
        atomicAdd(counter, my_count);
    }
}
```

### Pattern 2: Histogram

```cpp
#define NUM_BINS 256

__global__ void histogram(unsigned char *data, int *hist, int n) {
    __shared__ int local_hist[NUM_BINS];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    // Build local histogram
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge to global
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&hist[i], local_hist[i]);
        }
    }
}
```

### Pattern 3: Work Queue (Producer-Consumer)

```cpp
struct WorkQueue {
    int *tasks;
    int *head;  // Next task to produce
    int *tail;  // Next task to consume
    int capacity;
};

// Producer
__device__ void enqueue(WorkQueue *q, int task) {
    int pos = atomicAdd(q->head, 1) % q->capacity;
    q->tasks[pos] = task;
}

// Consumer
__device__ int dequeue(WorkQueue *q) {
    int pos = atomicAdd(q->tail, 1) % q->capacity;
    return q->tasks[pos];
}

__global__ void producerConsumer(WorkQueue *q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Produce
    if (idx % 2 == 0) {
        int task = compute_task();
        enqueue(q, task);
    }
    
    __syncthreads();
    
    // Consume
    if (idx % 2 == 1) {
        int task = dequeue(q);
        process_task(task);
    }
}
```

### Pattern 4: Finding Global Max/Min

```cpp
__global__ void findMaxOptimized(float *data, float *result, int n) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and reduce in shared memory
    shared[tid] = (idx < n) ? data[idx] : -INFINITY;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        float val = shared[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        
        if (tid == 0) {
            atomicMax((int*)result, __float_as_int(val));
        }
    }
}
```

### Pattern 5: Parallel Prefix Sum (Scan)

```cpp
// Inclusive scan within block
__device__ int blockScan(int val) {
    __shared__ int temp[256];
    
    int tid = threadIdx.x;
    temp[tid] = val;
    __syncthreads();
    
    // Up-sweep (reduce)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx + stride < blockDim.x) {
            temp[idx + stride] += temp[idx];
        }
        __syncthreads();
    }
    
    return temp[tid];
}
```

---

## Debugging and Troubleshooting {#debugging}

### Common Issues

#### **Issue 1: Race Condition**

**Symptoms:**
- Results vary between runs
- Incorrect output
- Works with fewer threads, fails with more

**Example:**
```cpp
// Bug
int temp = *shared_counter;
temp++;
*shared_counter = temp;

// Fix
atomicAdd(shared_counter, 1);
```

**Detection:**
```bash
# Run with cuda-memcheck
cuda-memcheck --tool racecheck ./myprogram

# Look for:
# "Race reported between Write access and Read access"
```

#### **Issue 2: Deadlock**

**Symptoms:**
- Program hangs
- GPU unresponsive
- Needs hard reset

**Common Causes:**
```cpp
// Conditional __syncthreads()
if (threadIdx.x < 128) {
    __syncthreads();  // ← Only some threads reach
}

// Spinlock held by descheduled thread
lock(&mutex);
// GPU scheduler switches to different warp
// Current warp never finishes, others spin forever
```

**Prevention:**
```cpp
// Always unconditional barriers
__syncthreads();
if (threadIdx.x < 128) {
    // Work
}

// Avoid locks entirely - use atomics!
atomicAdd(&counter, 1);
```

#### **Issue 3: Warp Divergence**

**Symptoms:**
- Much slower than expected
- Low occupancy
- Threads idle

**Detection:**
```bash
# Profile with Nsight Compute
ncu --metrics smsp__sass_branch_targets_threads_divergent ./myprogram

# Look for high divergence percentage
```

**Example:**
```cpp
// High divergence
if (threadIdx.x % 2 == 0) {
    lock(&mutex);  // Half threads spin
    // Critical section
    unlock(&mutex);
}

// Low divergence
atomicAdd(&counter, 1);  // All threads make progress
```

#### **Issue 4: Atomic Contention**

**Symptoms:**
- Slow despite using atomics
- GPU underutilized
- Long stalls

**Detection:**
```bash
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom ./myprogram

# High atomic traffic = contention
```

**Fix:**
```cpp
// Before: High contention
atomicAdd(&global_counter, 1);

// After: Staged atomics
__shared__ int local_counter;
atomicAdd(&local_counter, 1);
__syncthreads();
if (threadIdx.x == 0) {
    atomicAdd(&global_counter, local_counter);
}
```

### Debugging Tools

#### **CUDA-MEMCHECK**

```bash
# Race detection
cuda-memcheck --tool racecheck ./program

# Memory errors
cuda-memcheck --tool memcheck ./program

# Synchronization errors
cuda-memcheck --tool synccheck ./program
```

#### **CUDA-GDB**

```bash
# Debug kernel
cuda-gdb ./program

(cuda-gdb) break myKernel
(cuda-gdb) run
(cuda-gdb) cuda thread
(cuda-gdb) print variable
```

#### **Nsight Compute**

```bash
# Profile atomics
ncu --metrics atomic ./program

# Check occupancy
ncu --metrics sm_efficiency,achieved_occupancy ./program

# Full analysis
ncu --set full ./program
```

### Verification Strategies

```cpp
// 1. Sequential verification
void verifyResults(int *gpu_result, int *cpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_result[i]) {
            printf("Mismatch at %d: GPU=%d, CPU=%d\n",
                   i, gpu_result[i], cpu_result[i]);
        }
    }
}

// 2. Consistency checks
__global__ void checkInvariants(int *data) {
    // Example: Sum should equal expected
    __shared__ int sum;
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    
    atomicAdd(&sum, data[threadIdx.x]);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        assert(sum == EXPECTED_SUM);
    }
}

// 3. Stress testing
// Run with maximum threads
// Run many iterations
// Check for consistency across runs
```

---

## Summary

### Key Takeaways

```
╔══════════════════════════════════════════════════════════════╗
║                  CUDA SYNCHRONIZATION RULES                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Atomics > Locks (always!)                                ║
║  2. Warp shuffles > Shared memory > Global atomics           ║
║  3. __syncthreads() only within blocks                       ║
║  4. Multiple kernels for grid-level sync                     ║
║  5. Design lock-free from the start                          ║
║  6. Stage through shared memory to reduce contention         ║
║  7. Profile before optimizing                                ║
║  8. When in doubt, avoid locks!                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Performance Hierarchy

```
Fastest → Slowest:
─────────────────
1. No synchronization (if safe)
2. Warp-level primitives (__shfl_sync)
3. Shared memory atomics
4. Global memory atomics (staged)
5. Global memory atomics (direct)
6. __syncthreads() (necessary overhead)
7. Multiple kernel launches (clean sync)
8. Cooperative groups (grid sync)
9. Spinlocks (AVOID!)
10. Complex locks (NEVER USE!)
```

### Final Recommendation

**For 99% of use cases, this is all you need:**

```cpp
// Counters, sums
atomicAdd(&counter, 1);

// Max/min
atomicMax(&maximum, value);

// Histogram
atomicAdd(&histogram[bin], 1);

// Shared memory coordination
__syncthreads();

// Warp-level reduction
int sum = warpReduceSum(value);

// Grid-level phases
kernel1<<<...>>>(data);
cudaDeviceSynchronize();
kernel2<<<...>>>(data);
```

**Remember: Good GPU algorithms are fundamentally lock-free!**

---

## Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Cooperative Groups Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [CUDA Atomic Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)

---

**File**: [`examples/gpu_locks_and_synchronization.cu`](examples/gpu_locks_and_synchronization.cu) contains runnable examples of all concepts!

**Build & run**: `cd examples && make gpu_locks_and_synchronization && ./gpu_locks_and_synchronization`

← Back to [README](README.md) · Next: [13 — Streams & concurrency →](13_streams_and_concurrency.md)

