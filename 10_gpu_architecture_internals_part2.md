# GPU Architecture Internals: Part 2
## Continuation: Memory, Interconnects, and Evolution

*This is a continuation of [Part 1](10_gpu_architecture_internals.md)*

---

## Cache Hierarchy Details

### L1 Data Cache Structure

```
┌──────────────────────────────────────────────────────────────────┐
│               L1 DATA CACHE ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Unified with Shared Memory (128 KB total Ampere)                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  CACHE ORGANIZATION:                                   │      │
│  │  • Line size: 128 bytes                                │      │
│  │  • Associativity: 4-way set associative                │      │
│  │  • Write policy: Write-back, write-allocate            │      │
│  │  • Replacement: LRU (Least Recently Used)              │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  ADDRESS BREAKDOWN (128-byte line):                               │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                                                         │      │
│  │  Virtual Address (64-bit):                             │      │
│  │  ┌────────────┬─────────────┬──────────┬──────────┐   │      │
│  │  │    Tag     │  Set Index  │  Offset  │  Byte    │   │      │
│  │  │  (bits)    │   (bits)    │ (4 bits) │ (3 bits) │   │      │
│  │  └────────────┴─────────────┴──────────┴──────────┘   │      │
│  │       │             │             │           │        │      │
│  │       │             │             │           └─> Within 8-byte word
│  │       │             │             └─> Within cache line (16 words)
│  │       │             └─> Selects cache set
│  │       └─> Compared for hit/miss                        │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  4-WAY SET-ASSOCIATIVE STRUCTURE:                                 │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Set 0:  [Way 0][Way 1][Way 2][Way 3]                 │      │
│  │  Set 1:  [Way 0][Way 1][Way 2][Way 3]                 │      │
│  │  Set 2:  [Way 0][Way 1][Way 2][Way 3]                 │      │
│  │  ...                                                    │      │
│  │  Set N:  [Way 0][Way 1][Way 2][Way 3]                 │      │
│  │                                                         │      │
│  │  Each Way:                                              │      │
│  │  ┌──────────────────────────────────────────────┐     │      │
│  │  │ Valid │ Dirty │ Tag │ Data (128 bytes)     │     │      │
│  │  │  (1b) │  (1b) │(...)│                       │     │      │
│  │  └──────────────────────────────────────────────┘     │      │
│  │                                                         │      │
│  │  Lookup Process:                                        │      │
│  │  1. Extract set index from address                     │      │
│  │  2. Check all 4 ways in parallel                       │      │
│  │  3. Compare tags                                        │      │
│  │  4. Hit: Return data from matching way                 │      │
│  │  5. Miss: Evict LRU way, fetch from L2                 │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  CACHE COHERENCE:                                                 │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  L1 caches are NOT coherent across SMs!                │      │
│  │                                                         │      │
│  │  Implications:                                          │      │
│  │  • Reads may see stale data from other SMs             │      │
│  │  • Must use __threadfence_system() for coherence       │      │
│  │  • Atomics bypass L1 (go to L2)                        │      │
│  │  • L2 maintains coherence                              │      │
│  │                                                         │      │
│  │  Cache Control:                                         │      │
│  │  • Loads: Cached by default                            │      │
│  │  • Stores: Write-through to L2                         │      │
│  │  • Can use caching modifiers:                          │      │
│  │    - ld.ca  (cache all levels)                         │      │
│  │    - ld.cg  (cache at L2 only)                         │      │
│  │    - ld.cs  (streaming, bypass cache)                  │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  PERFORMANCE CHARACTERISTICS:                                     │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Hit Latency:     ~30 cycles                           │      │
│  │  Miss Penalty:    ~170 cycles (L2 access)              │      │
│  │  Bandwidth:       128 bytes/cycle per SM               │      │
│  │  Typical Hit Rate: 70-95% (workload dependent)         │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### L2 Cache Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                    L2 CACHE ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Chip-wide unified cache (6 MB on Ampere GA102)                      │
│                                                                       │
│  PARTITIONED STRUCTURE:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  L2 Cache Slices (matched to memory controllers):            │   │
│  │                                                               │   │
│  │  ┌────────┬────────┬────────┬────────┬────────┬────────┐    │   │
│  │  │Slice 0 │Slice 1 │Slice 2 │Slice 3 │Slice 4 │Slice 5 │    │   │
│  │  │ 768KB  │ 768KB  │ 768KB  │ 768KB  │ 768KB  │ 768KB  │    │   │
│  │  └───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘    │   │
│  │      │        │        │        │        │        │          │   │
│  │     MC0      MC1      MC2      MC3      MC4      MC5         │   │
│  │                                                               │   │
│  │  Each slice connects to one memory controller                │   │
│  │  Address interleaving for load balancing                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  CACHE PARAMETERS:                                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  • Total Size: 6 MB (Ampere), 40 MB (Hopper)                │   │
│  │  • Line Size: 128 bytes                                      │   │
│  │  • Associativity: 16-way set associative                     │   │
│  │  • Write Policy: Write-back                                  │   │
│  │  • Replacement: Approximated LRU with sector promotion       │   │
│  │  • ECC: Protected with SECDED (optional)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ADDRESS MAPPING (Interleaved):                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  Physical Address:                                            │   │
│  │  ┌─────────┬────────┬──────────┬──────────┬────────┐        │   │
│  │  │   Tag   │  Set   │  Slice   │  Line    │ Byte   │        │   │
│  │  │ (bits)  │ Index  │  Select  │  Offset  │ Offset │        │   │
│  │  └─────────┴────────┴──────────┴──────────┴────────┘        │   │
│  │      │         │         │           │         │             │   │
│  │      │         │         │           │         └─> 0-7 (8B)  │   │
│  │      │         │         │           └─> 0-15 (16 words)     │   │
│  │      │         │         └─> Slice 0-5                       │   │
│  │      │         └─> Set within slice                          │   │
│  │      └─> Tag comparison                                      │   │
│  │                                                               │   │
│  │  Hash function distributes addresses across slices           │   │
│  │  for uniform memory controller utilization                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  PER-SLICE STRUCTURE (768 KB):                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  Number of Sets = 768KB / (128 bytes × 16 ways) = 384 sets  │   │
│  │                                                               │   │
│  │  16-Way Set Organization:                                     │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ Set 0:                                             │     │   │
│  │  │ [W0][W1][W2][W3][W4][W5][W6][W7]....[W14][W15]   │     │   │
│  │  │                                                    │     │   │
│  │  │ Set 1:                                             │     │   │
│  │  │ [W0][W1][W2][W3][W4][W5][W6][W7]....[W14][W15]   │     │   │
│  │  │                                                    │     │   │
│  │  │ ...                                                │     │   │
│  │  │                                                    │     │   │
│  │  │ Set 383:                                           │     │   │
│  │  │ [W0][W1][W2][W3][W4][W5][W6][W7]....[W14][W15]   │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  Each Way Entry:                                              │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ [V][D][Tag][ECC][    Data: 128 bytes    ][ECC]   │     │   │
│  │  │  1b 1b (Xb)  Xb   (16 × 8-byte words)     Xb     │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │  V = Valid bit, D = Dirty bit                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ATOMIC OPERATIONS IN L2:                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  L2 cache handles all atomic operations:                     │   │
│  │                                                               │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ 1. Atomic request arrives at L2                   │     │   │
│  │  │ 2. Line locked (prevents other access)            │     │   │
│  │  │ 3. Read-modify-write in L2                        │     │   │
│  │  │ 4. Write back to memory (if needed)               │     │   │
│  │  │ 5. Release lock                                    │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  Atomic Unit per L2 slice:                                   │   │
│  │  • Handles atomicAdd, atomicCAS, etc.                        │   │
│  │  • Serializes conflicting atomics                            │   │
│  │  • Can process multiple non-conflicting atomics              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  CACHE RESIDENT FEATURE (Ampere):                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Allows pinning data in L2:                                  │   │
│  │                                                               │   │
│  │  • Reserve portion of L2 for specific data                   │   │
│  │  • Prevents eviction of critical data                        │   │
│  │  • Useful for:                                               │   │
│  │    - Frequently accessed lookup tables                       │   │
│  │    - Kernel parameters                                       │   │
│  │    - Shared data structures                                  │   │
│  │                                                               │   │
│  │  Usage: cudaMemAdvise() with cudaMemAdviseSetPreferredLocation
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  PERFORMANCE:                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Hit Latency:        ~200 cycles                             │   │
│  │  Miss Penalty:       ~200+ cycles (DRAM access)              │   │
│  │  Bandwidth:          ~1500 GB/s (internal)                   │   │
│  │  Typical Hit Rate:   60-80%                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Memory Controllers and Interfaces

### Memory Controller Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                 MEMORY CONTROLLER ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  GDDR6X MEMORY CONTROLLER (Per Controller):                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │         REQUEST QUEUE & ARBITER                   │     │   │
│  │  │  ┌──────────────────────────────────────────┐    │     │   │
│  │  │  │ From L2 Cache Slice                      │    │     │   │
│  │  │  │  • Read requests                          │    │     │   │
│  │  │  │  • Write requests                         │    │     │   │
│  │  │  │  • Priority levels                        │    │     │   │
│  │  │  │  • Atomic operations                      │    │     │   │
│  │  │  └──────────────────────────────────────────┘    │     │   │
│  │  │                      ↓                             │     │   │
│  │  │  ┌──────────────────────────────────────────┐    │     │   │
│  │  │  │ Request Scheduling                        │    │     │   │
│  │  │  │  • Bank conflict avoidance                │    │     │   │
│  │  │  │  • Row buffer hit optimization            │    │     │   │
│  │  │  │  • Read/write batching                    │    │     │   │
│  │  │  │  • Priority arbitration                   │    │     │   │
│  │  │  └──────────────────────────────────────────┘    │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                      ↓                                       │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │         DRAM COMMAND GENERATOR                    │     │   │
│  │  │  • Activate (open row)                             │     │   │
│  │  │  • Read / Write                                    │     │   │
│  │  │  • Precharge (close row)                           │     │   │
│  │  │  • Refresh                                         │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                      ↓                                       │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │            PHY (Physical Interface)                │     │   │
│  │  │  • Data bus: 32-bit per channel (×2 for 64-bit)   │     │   │
│  │  │  • Clock: Up to 21 Gbps (GDDR6X)                  │     │   │
│  │  │  • Signaling: PAM4 (4-level)                       │     │   │
│  │  │  • ECC encoding/decoding                           │     │   │
│  │  │  • Calibration and training                        │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                      ↕                                       │   │
│  │              [GDDR6X Memory Chip]                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  MEMORY ORGANIZATION:                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  GDDR6X Chip Structure:                                       │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │  Chip (2 GB typical)                               │     │   │
│  │  │                                                     │     │   │
│  │  │  ┌──────────┬──────────┬──────────┬──────────┐    │     │   │
│  │  │  │ Channel 0│ Channel 1│ Channel 2│ Channel 3│    │     │   │
│  │  │  │  (512MB) │  (512MB) │  (512MB) │  (512MB) │    │     │   │
│  │  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘    │     │   │
│  │  │       │          │          │          │           │     │   │
│  │  │  Each Channel has:                                  │     │   │
│  │  │  ┌──────────────────────────────────────────┐     │     │   │
│  │  │  │ • 16 Banks                                │     │     │   │
│  │  │  │ • Each Bank: 32K rows × 1024 columns     │     │     │   │
│  │  │  │ • Row size: 8 KB                          │     │     │   │
│  │  │  │ • Column access: 32 bytes                 │     │     │   │
│  │  │  └──────────────────────────────────────────┘     │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  Address Mapping (Example):                                   │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ Physical Address Bits:                             │     │   │
│  │  │ [Channel][Bank][Row][Column][Byte]                │     │   │
│  │  │    2b      4b    15b    10b     5b                 │     │   │
│  │  │                                                     │     │   │
│  │  │ Interleaving for parallel access                   │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ROW BUFFER MANAGEMENT:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  Each bank has a row buffer (8 KB):                          │   │
│  │                                                               │   │
│  │  ROW HIT (Fast):                                              │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ 1. Request arrives                                 │     │   │
│  │  │ 2. Requested row already in buffer                 │     │   │
│  │  │ 3. Read/write directly from buffer                 │     │   │
│  │  │ Latency: ~40 ns                                    │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  ROW MISS (Slow):                                             │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ 1. Request arrives                                 │     │   │
│  │  │ 2. Different row in buffer (conflict)              │     │   │
│  │  │ 3. PRECHARGE (close current row) ~20 ns            │     │   │
│  │  │ 4. ACTIVATE (open new row) ~20 ns                  │     │   │
│  │  │ 5. READ/WRITE ~20 ns                               │     │   │
│  │  │ Total Latency: ~60+ ns                             │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  Optimization: Controller tries to keep rows open              │   │
│  │  and schedule requests to maximize row buffer hits            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  BANDWIDTH CALCULATION:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  GDDR6X Example (RTX 3090):                                  │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ • Memory Clock: 19.5 Gbps (effective)              │     │   │
│  │  │ • Bus Width: 384-bit (12 × 32-bit channels)        │     │   │
│  │  │ • Theoretical Bandwidth:                           │     │   │
│  │  │   = 19.5 Gbps × 384 bits / 8 bits/byte             │     │   │
│  │  │   = 936 GB/s                                       │     │   │
│  │  │                                                     │     │   │
│  │  │ • Practical Bandwidth: ~70-80% of theoretical      │     │   │
│  │  │   = ~700-750 GB/s achievable                       │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  HBM2e Example (A100):                                        │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ • Memory Clock: 3.2 Gbps (per pin)                 │     │   │
│  │  │ • Bus Width: 5120-bit (8 stacks × 1024-bit/2)      │     │   │
│  │  │ • Theoretical Bandwidth:                           │     │   │
│  │  │   = 3.2 Gbps × 5120 bits / 8 bits/byte             │     │   │
│  │  │   = 2039 GB/s (~2 TB/s)                            │     │   │
│  │  │                                                     │     │   │
│  │  │ • Practical Bandwidth: ~80-85%                     │     │   │
│  │  │   = ~1600-1700 GB/s achievable                     │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### HBM vs GDDR Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                     HBM vs GDDR COMPARISON                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  GDDR6X (Discrete Memory):                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  GPU Package          Memory Chips (separate)                │   │
│  │  ┌──────────┐        ┌────┐┌────┐┌────┐┌────┐              │   │
│  │  │          │        │GDDR││GDDR││GDDR││GDDR│              │   │
│  │  │   GPU    │◄──────►│ 6X ││ 6X ││ 6X ││ 6X │              │   │
│  │  │   Die    │   PCB  │Chip││Chip││Chip││Chip│              │   │
│  │  │          │  trace │    ││    ││    ││    │              │   │
│  │  └──────────┘        └────┘└────┘└────┘└────┘              │   │
│  │                                                               │   │
│  │  Characteristics:                                             │   │
│  │  • High clock speed (19+ Gbps)                               │   │
│  │  • Narrow bus per chip (32-bit)                              │   │
│  │  • Long traces → higher power, latency                       │   │
│  │  • Lower cost                                                 │   │
│  │  • Easier to manufacture                                     │   │
│  │  • Typical: 320-512 bit total bus width                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  HBM2e (Stacked Memory):                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  ┌────────────────────────────────────────────────┐          │   │
│  │  │  Memory Stack (HBM)                            │          │   │
│  │  │  ┌──────────────────────────────────────┐      │          │   │
│  │  │  │ DRAM Die 7 (Top)                     │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 6                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 5                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 4                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 3                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 2                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ DRAM Die 1                           │      │          │   │
│  │  │  ├──────────────────────────────────────┤      │          │   │
│  │  │  │ Base Die (Logic)                     │      │          │   │
│  │  │  └──────────────────────────────────────┘      │          │   │
│  │  │          │ TSV (Through-Silicon Vias)          │          │   │
│  │  └──────────┼──────────────────────────────────────┘          │   │
│  │             ↓                                                 │   │
│  │  ┌─────────────────────────────────────────┐                 │   │
│  │  │         GPU Die                          │                 │   │
│  │  │  (Connected via Silicon Interposer)      │                 │   │
│  │  └─────────────────────────────────────────┘                 │   │
│  │                                                               │   │
│  │  Characteristics:                                             │   │
│  │  • Lower clock speed (2-3.2 Gbps)                            │   │
│  │  • Very wide bus per stack (1024-bit)                        │   │
│  │  • Short paths → lower power, latency                        │   │
│  │  • Higher cost                                                │   │
│  │  • More complex to manufacture                               │   │
│  │  • Typical: 4096-5120 bit total bus width                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  COMPARISON TABLE:                                                    │
│  ┌───────────────┬──────────────┬──────────────┬──────────────┐    │
│  │ Feature       │ GDDR6        │ GDDR6X       │ HBM2e        │    │
│  ├───────────────┼──────────────┼──────────────┼──────────────┤    │
│  │ Speed/Pin     │ 14-16 Gbps   │ 19-21 Gbps   │ 2.4-3.6 Gbps │    │
│  │ Bus Width     │ 384-512 bit  │ 384-512 bit  │ 4096-6144 bit│    │
│  │ Bandwidth     │ ~750 GB/s    │ ~950 GB/s    │ ~1600 GB/s   │    │
│  │ Power         │ Medium       │ High         │ Lower        │    │
│  │ Latency       │ ~120 ns      │ ~100 ns      │ ~80 ns       │    │
│  │ Cost          │ Low          │ Medium       │ High         │    │
│  │ Capacity      │ 8-24 GB      │ 10-24 GB     │ 40-80 GB     │    │
│  │ Use Case      │ Consumer GPU │ High-end GPU │ Data Center  │    │
│  └───────────────┴──────────────┴──────────────┴──────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Interconnect Architecture

### NVLink Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     NVLINK INTERCONNECT                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  NVLink is NVIDIA's high-speed GPU-to-GPU interconnect               │
│                                                                       │
│  NVLINK 3.0 (Ampere) ARCHITECTURE:                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  GPU 0                                  GPU 1                 │   │
│  │  ┌──────────────┐                      ┌──────────────┐      │   │
│  │  │              │◄────── Link 0 ──────►│              │      │   │
│  │  │              │◄────── Link 1 ──────►│              │      │   │
│  │  │     SM       │◄────── Link 2 ──────►│     SM       │      │   │
│  │  │   Array      │◄────── Link 3 ──────►│   Array      │      │   │
│  │  │              │◄────── Link 4 ──────►│              │      │   │
│  │  │              │◄────── Link 5 ──────►│              │      │   │
│  │  └──────────────┘                      └──────────────┘      │   │
│  │                                                               │   │
│  │  Each Link:                                                   │   │
│  │  • Bidirectional                                              │   │
│  │  • 25 GB/s per direction                                      │   │
│  │  • 50 GB/s total per link                                     │   │
│  │  • 6 links = 300 GB/s total bidirectional bandwidth           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  NVLINK SWITCH (NVSwitch for multi-GPU):                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │              GPU 0          GPU 1          GPU 2              │   │
│  │                │              │              │                │   │
│  │                └──────┬───────┴──────┬───────┘                │   │
│  │                       │              │                        │   │
│  │                  ┌────▼──────────────▼────┐                   │   │
│  │                  │                        │                   │   │
│  │                  │      NVSwitch          │                   │   │
│  │                  │   (Crossbar Switch)    │                   │   │
│  │                  │                        │                   │   │
│  │                  └────┬──────────────┬────┘                   │   │
│  │                       │              │                        │   │
│  │                ┌──────┴───────┬──────┴───────┐                │   │
│  │                │              │              │                │   │
│  │              GPU 3          GPU 4          GPU 5              │   │
│  │                                                               │   │
│  │  • All-to-all connectivity                                    │   │
│  │  • 900 GB/s bisection bandwidth (NVSwitch 2.0)                │   │
│  │  • Used in DGX systems                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  NVLINK PROTOCOL LAYERS:                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  Transaction Layer (TL)                             │    │   │
│  │  │  • Packet formation                                 │    │   │
│  │  │  • Flow control                                     │    │   │
│  │  │  • Error detection & retry                          │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                        ↕                                      │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  Data Link Layer (DLL)                              │    │   │
│  │  │  • Credit-based flow control                        │    │   │
│  │  │  • CRC error checking                               │    │   │
│  │  │  • Link training                                    │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                        ↕                                      │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  Physical Layer (PHY)                               │    │   │
│  │  │  • High-speed SerDes (Serializer/Deserializer)     │    │   │
│  │  │  • 25 Gbps per lane (NVLink 3.0)                    │    │   │
│  │  │  • 8 differential pairs per link                    │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  CAPABILITIES:                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  • Direct GPU-to-GPU memory access                           │   │
│  │  • Cache coherence between GPUs                              │   │
│  │  • Atomic operations across link                             │   │
│  │  • RDMA (Remote Direct Memory Access)                        │   │
│  │  • GPU peer-to-peer (P2P) memory copy                        │   │
│  │  • Lower latency than PCIe (~5x faster)                      │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  NVLINK vs PCIe:                                                      │
│  ┌─────────────────┬──────────────┬───────────────┬──────────────┐  │
│  │ Feature         │ PCIe 4.0     │ PCIe 5.0      │ NVLink 3.0   │  │
│  ├─────────────────┼──────────────┼───────────────┼──────────────┤  │
│  │ Bandwidth       │ 64 GB/s      │ 128 GB/s      │ 600 GB/s     │  │
│  │ (bidirectional) │ (x16 lanes)  │ (x16 lanes)   │ (12 links)   │  │
│  │ Latency         │ ~1-2 μs      │ ~1-2 μs       │ ~0.5 μs      │  │
│  │ Cache Coherence │ No           │ No            │ Yes          │  │
│  │ P2P Atomics     │ Limited      │ Limited       │ Full         │  │
│  │ Use Case        │ Host-Device  │ Host-Device   │ GPU-GPU      │  │
│  └─────────────────┴──────────────┴───────────────┴──────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### PCIe Interface

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PCIe INTERFACE                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PCIe connects GPU to CPU and system memory                          │
│                                                                       │
│  TYPICAL CONFIGURATION (PCIe 4.0 x16):                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  ┌────────────┐                          ┌──────────────┐    │   │
│  │  │    CPU     │                          │     GPU      │    │   │
│  │  │            │                          │              │    │   │
│  │  │  ┌──────┐  │                          │  ┌────────┐  │    │   │
│  │  │  │ PCIe │◄─┼──────── x16 Link ───────►│  │ PCIe   │  │    │   │
│  │  │  │ Root │  │    (16 lanes × 2 GB/s)   │  │ Endpt  │  │    │   │
│  │  │  │Complex│  │    = 32 GB/s each way    │  │        │  │    │   │
│  │  │  └──────┘  │                          │  └────────┘  │    │   │
│  │  │            │                          │              │    │   │
│  │  └────────────┘                          └──────────────┘    │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  PCIe LANE STRUCTURE:                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  Each Lane (Bidirectional):                                  │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │  TX Pair: ──────────────────────────────────►      │     │   │
│  │  │           Differential signaling                    │     │   │
│  │  │  RX Pair: ◄──────────────────────────────────      │     │   │
│  │  │           Differential signaling                    │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  x16 Link = 16 differential pairs each direction             │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  BANDWIDTH BY GENERATION:                                             │
│  ┌─────────────┬──────────────┬─────────────┬─────────────────┐    │
│  │ Generation  │ Per Lane     │ x16 Total   │ Bidirectional   │    │
│  ├─────────────┼──────────────┼─────────────┼─────────────────┤    │
│  │ PCIe 3.0    │ 985 MB/s     │ ~16 GB/s    │ ~32 GB/s        │    │
│  │ PCIe 4.0    │ 1.969 GB/s   │ ~32 GB/s    │ ~64 GB/s        │    │
│  │ PCIe 5.0    │ 3.938 GB/s   │ ~64 GB/s    │ ~128 GB/s       │    │
│  │ PCIe 6.0    │ 7.5 GB/s     │ ~120 GB/s   │ ~240 GB/s       │    │
│  └─────────────┴──────────────┴─────────────┴─────────────────┘    │
│                                                                       │
│  TRANSFER TYPES:                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  1. Programmed I/O (PIO):                                     │   │
│  │     • CPU writes to GPU memory                               │   │
│  │     • Used for small transfers                               │   │
│  │     • CPU overhead                                           │   │
│  │                                                               │   │
│  │  2. DMA (Direct Memory Access):                               │   │
│  │     • GPU initiates transfer                                 │   │
│  │     • No CPU intervention                                    │   │
│  │     • Efficient for large transfers                          │   │
│  │                                                               │   │
│  │  3. Peer-to-Peer (P2P):                                       │   │
│  │     • GPU-to-GPU transfer via PCIe                           │   │
│  │     • Bypasses system memory                                 │   │
│  │     • Requires BAR (Base Address Register) mapping           │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  LATENCY CHARACTERISTICS:                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  • Base latency: ~1-2 μs                                      │   │
│  │  • Small transfer (<4KB): ~2-3 μs                             │   │
│  │  • Large transfer: Latency + (Size / Bandwidth)               │   │
│  │                                                               │   │
│  │  Factors affecting latency:                                   │   │
│  │  • TLP (Transaction Layer Packet) overhead                   │   │
│  │  • CPU-GPU distance (PCIe hops)                              │   │
│  │  • Memory type (pageable vs pinned)                          │   │
│  │  • System load                                               │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Evolution

### Generational Comparison

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 NVIDIA GPU ARCHITECTURE EVOLUTION                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  KEPLER (2012) → MAXWELL (2014) → PASCAL (2016) → VOLTA (2017)              │
│      → TURING (2018) → AMPERE (2020) → ADA (2022) → HOPPER (2022)           │
│                                                                               │
├───────────────┬────────┬────────┬────────┬────────┬────────┬────────┬───────┤
│ Feature       │ Kepler │Maxwell │ Pascal │ Volta  │ Ampere │  Ada   │Hopper │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Process Node  │ 28nm   │ 28nm   │ 16nm   │ 12nm   │  8nm   │  4nm   │  4nm  │
│ (TSMC)        │        │        │        │ (TSMC) │(Samsung)│ (TSMC) │(TSMC) │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ SM Count      │  15    │  24    │  56    │  80    │  108   │  128   │  132  │
│ (Flagship)    │ (K80)  │(M40)   │(P100)  │(V100)  │(A100)  │(4090)  │(H100) │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ CUDA Cores/SM │  192   │  128   │  64    │  64    │  64    │  128   │  128  │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ FP64 Rate     │ 1/3    │ 1/32   │ 1/2    │ 1/2    │ 1/64   │ 1/64   │ 1/2   │
│ (vs FP32)     │        │        │(P100)  │        │        │        │       │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Tensor Cores  │  No    │  No    │  No    │  Yes   │  Yes   │  Yes   │  Yes  │
│               │        │        │        │(1st gen)│(3rd gen)│(4th gen)│(4th)│
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ RT Cores      │  No    │  No    │  No    │  No    │  No    │  Yes   │  No   │
│               │        │        │        │        │        │(3rd gen)│       │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Shared Mem/SM │ 48KB   │  96KB  │  64KB  │  96KB  │ 164KB  │ 128KB  │ 228KB │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ L2 Cache      │ 1.5MB  │  3MB   │  4MB   │  6MB   │  40MB  │  72MB  │  50MB │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Memory Type   │ GDDR5  │ GDDR5  │ HBM2   │ HBM2   │ HBM2e  │ GDDR6X │ HBM3  │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Memory BW     │ 480    │ 288    │  732   │  900   │ 1935   │ 1008   │ 3000  │
│ (GB/s)        │        │        │        │        │        │        │       │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ NVLink        │  No    │  No    │  Yes   │  Yes   │  Yes   │  No    │  Yes  │
│               │        │        │ (1.0)  │ (2.0)  │ (3.0)  │        │ (4.0) │
├───────────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ TDP (Watts)   │ 235    │ 250    │  300   │  300   │  400   │  450   │  700  │
└───────────────┴────────┴────────┴────────┴────────┴────────┴────────┴───────┘
```

### Key Architectural Improvements

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MAJOR ARCHITECTURAL INNOVATIONS                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  MAXWELL (2014):                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • New SM design: 128 CUDA cores per SM                             │    │
│  │  • Better power efficiency (~2x per watt vs Kepler)                 │    │
│  │  • Improved scheduler: 4 warp schedulers                             │    │
│  │  • Shared memory/L1 cache unified and configurable                  │    │
│  │  • Dynamic parallelism improvements                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  PASCAL (2016):                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 16nm FinFET process (first)                                      │    │
│  │  • HBM2 memory (first GPU)                                          │    │
│  │  • NVLink interconnect                                              │    │
│  │  • Unified memory improvements                                       │    │
│  │  • FP16 performance: 2x FP32 rate                                   │    │
│  │  • Preemption: Instruction-level                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  VOLTA (2017):                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Tensor Cores (first generation)                                  │    │
│  │    - 640 Tensor ops/clock per SM                                    │    │
│  │    - Mixed precision: FP16 input, FP32 accumulate                   │    │
│  │  • Independent thread scheduling                                     │    │
│  │  • L1 cache + shared memory combined (128KB)                        │    │
│  │  • Improved FP64 performance (1/2 FP32 rate)                        │    │
│  │  • HBM2 with ECC                                                    │    │
│  │  • NVLink 2.0 (300 GB/s)                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  TURING (2018):                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • RT Cores (ray tracing acceleration)                              │    │
│  │  • Tensor Cores 2nd gen                                             │    │
│  │    - INT8/INT4/Binary precision support                             │    │
│  │  • Mesh shading                                                     │    │
│  │  • Variable rate shading                                            │    │
│  │  • GDDR6 memory                                                     │    │
│  │  • Concurrent FP32 and INT32 execution                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  AMPERE (2020):                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Tensor Cores 3rd gen                                             │    │
│  │    - Sparsity acceleration (2:4 structured sparsity)                │    │
│  │    - TF32 precision (NVIDIA TensorFloat-32)                         │    │
│  │    - BF16 (Brain Float 16) support                                  │    │
│  │  • 2x FP32 throughput (dual FP32 datapaths)                         │    │
│  │  • Multi-Instance GPU (MIG) - up to 7 instances                     │    │
│  │  • 3rd gen NVLink (600 GB/s)                                        │    │
│  │  • Huge L2 cache (40 MB in A100)                                    │    │
│  │  • Asynchronous copy (DMA between global and shared memory)         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ADA LOVELACE (2022):                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 4nm process (TSMC)                                               │    │
│  │  • Tensor Cores 4th gen                                             │    │
│  │    - FP8 support (Transformer Engine)                               │    │
│  │    - 2x throughput vs Ampere                                        │    │
│  │  • RT Cores 3rd gen                                                 │    │
│  │    - 2x ray-triangle intersection throughput                        │    │
│  │    - Opacity Micromap Engine (OMM)                                  │    │
│  │    - Displaced Micro-Mesh Engine (DMM)                              │    │
│  │  • GDDR6X memory (up to 24 GB)                                      │    │
│  │  • Ada Optical Flow Accelerator                                     │    │
│  │  • AV1 encoding/decoding                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  HOPPER (2022):                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Tensor Cores 4th gen with Transformer Engine                     │    │
│  │    - FP8 with FP32 accumulate                                       │    │
│  │    - 6x faster transformer training vs Ampere                       │    │
│  │  • HBM3 memory (3 TB/s bandwidth)                                   │    │
│  │  • NVLink 4.0 (900 GB/s)                                            │    │
│  │  • PCIe 5.0 support                                                 │    │
│  │  • Thread Block Clusters (new level in hierarchy)                   │    │
│  │  • Distributed Shared Memory                                        │    │
│  │  • Tensor Memory Accelerator (TMA)                                  │    │
│  │  • Dynamic Programming Instructions                                 │    │
│  │  • Confidential Computing support                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Roofline Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                      ROOFLINE PERFORMANCE MODEL                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Performance                                                          │
│  (GFLOPS)                                                             │
│      │                                                                │
│      │                                                                │
│ 1000 │                  ╱────────────────────  Peak Compute          │
│      │                ╱                         (Compute Bound)      │
│      │              ╱                                                 │
│  500 │            ╱                                                   │
│      │          ╱                                                     │
│      │        ╱                                                       │
│  100 │      ╱  Memory Bandwidth Roof                                 │
│      │    ╱    (Memory Bound)                                        │
│   50 │  ╱                                                             │
│      │╱                                                               │
│    0 └──────┬─────┬─────┬─────┬─────┬─────┬──────                   │
│            0.1   0.5    1     5    10    50   100                    │
│                                                                       │
│                  Arithmetic Intensity                                 │
│                  (FLOPs / Byte)                                       │
│                                                                       │
│  INTERPRETATION:                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                                                             │     │
│  │  Ridge Point = Peak Compute / Peak Bandwidth               │     │
│  │                                                             │     │
│  │  Example (A100):                                            │     │
│  │  • Peak FP32: 19.5 TFLOPS                                  │     │
│  │  • Peak Memory BW: 1935 GB/s                               │     │
│  │  • Ridge Point: 19500 / 1935 ≈ 10 FLOP/byte               │     │
│  │                                                             │     │
│  │  If your kernel has:                                        │     │
│  │  • AI < 10: Memory bound → optimize memory access          │     │
│  │  • AI > 10: Compute bound → optimize compute               │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ACTUAL KERNEL PLACEMENT:                                             │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Kernel Type          │ AI (FLOP/byte) │ Bottleneck        │     │
│  ├──────────────────────┼────────────────┼───────────────────┤     │
│  │  Vector Add          │ 0.083          │ Memory BW         │     │
│  │  Vector Multiply     │ 0.25           │ Memory BW         │     │
│  │  Dot Product         │ 0.5            │ Memory BW         │     │
│  │  SAXPY               │ 0.33           │ Memory BW         │     │
│  │  Matrix Multiply     │ 13.3 (N=1024)  │ Compute           │     │
│  │  (naive)             │                │                   │     │
│  │  Matrix Multiply     │ 42.7 (N=1024)  │ Compute           │     │
│  │  (optimized)         │                │                   │     │
│  │  FFT                 │ 2.5            │ Memory BW         │     │
│  │  Conv2D              │ 8-50           │ Mixed             │     │
│  └──────────────────────┴────────────────┴───────────────────┘     │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Latency Numbers

```
┌──────────────────────────────────────────────────────────────────────┐
│               LATENCY CHARACTERISTICS (Approximate)                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ARITHMETIC OPERATIONS:                                               │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Operation                    │ Latency (cycles)            │     │
│  ├──────────────────────────────┼─────────────────────────────┤     │
│  │  FP32 Add/Multiply            │ 4                           │     │
│  │  FP32 FMA (Fused Mul-Add)     │ 4                           │     │
│  │  FP32 Division                │ 16-20                       │     │
│  │  FP32 Square Root             │ 16-20                       │     │
│  │  FP32 Transcendental (sin)    │ 16-20                       │     │
│  │  INT32 Add/Logical            │ 4                           │     │
│  │  Tensor Core (4×4×4 MMA)      │ 8                           │     │
│  └──────────────────────────────┴─────────────────────────────┘     │
│                                                                       │
│  MEMORY ACCESS:                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Memory Type                  │ Latency (cycles) │ Latency  │     │
│  ├──────────────────────────────┼──────────────────┼──────────┤     │
│  │  Register                     │ 1                │ 0.5 ns   │     │
│  │  Shared Memory (no conflict)  │ 20-30            │ 10-15 ns │     │
│  │  Shared Memory (2-way conflict)│ 40-60           │ 20-30 ns │     │
│  │  L1 Cache Hit                 │ 30               │ 15 ns    │     │
│  │  Texture Cache Hit            │ 100              │ 50 ns    │     │
│  │  L2 Cache Hit                 │ 200              │ 100 ns   │     │
│  │  Global Memory (L2 miss)      │ 400-800          │ 200-400ns│     │
│  │  Atomic Operation (L2)        │ 200-400          │ 100-200ns│     │
│  │  Host Memory (PCIe)           │ 2000+            │ 1+ μs    │     │
│  └──────────────────────────────┴──────────────────┴──────────┘     │
│                                                                       │
│  CONTROL FLOW:                                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Operation                    │ Latency                     │     │
│  ├──────────────────────────────┼─────────────────────────────┤     │
│  │  Branch (predicted correctly) │ 0 (pipelined)               │     │
│  │  Branch (mispredicted)        │ 20-40 cycles                │     │
│  │  __syncthreads()              │ 20-30 cycles                │     │
│  │  __threadfence()              │ 100-200 cycles              │     │
│  │  __threadfence_system()       │ 200-400 cycles              │     │
│  │  Kernel Launch                │ 5-10 μs                     │     │
│  └──────────────────────────────┴─────────────────────────────┘     │
│                                                                       │
│  BANDWIDTH-LIMITED OPERATIONS:                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  For bandwidth-limited kernels:                            │     │
│  │                                                             │     │
│  │  Time = Data_Size / Effective_Bandwidth                    │     │
│  │                                                             │     │
│  │  Example: Copy 1 GB                                         │     │
│  │  • Peak Bandwidth: 760 GB/s (RTX 3090)                     │     │
│  │  • Achievable: ~600 GB/s (80% efficiency)                  │     │
│  │  • Time: 1 GB / 600 GB/s ≈ 1.67 ms                         │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This deep dive covered the internal architecture of NVIDIA GPUs including:

✓ **Die-level organization** - GPCs, TPCs, SMs, memory controllers  
✓ **SM microarchitecture** - Processing blocks, execution units, pipelines  
✓ **Memory subsystem** - Register files, shared memory banks, cache hierarchy  
✓ **Warp scheduling** - Instruction dispatch, latency hiding, divergence handling  
✓ **Memory controllers** - GDDR6X, HBM architecture, row buffer management  
✓ **Interconnects** - NVLink, PCIe interfaces  
✓ **Architectural evolution** - From Kepler to Hopper  
✓ **Performance models** - Roofline, latency numbers  

### Key Takeaways for Advanced Users

1. **Hierarchy Matters**: Understand the GPU hierarchy from die → GPC → SM → processing block
2. **Memory is King**: Memory bandwidth often limits performance more than compute
3. **Warp-Level Thinking**: Think in terms of warps (32 threads) for optimal performance
4. **Bank Conflicts**: Shared memory banking can make 32x performance difference
5. **Latency Hiding**: Use enough concurrent warps to hide memory latency
6. **Architecture-Specific**: Optimize for your target GPU architecture

### Further Reading

- **NVIDIA Whitepapers**: Each architecture has detailed whitepapers
- **PTX ISA Documentation**: Low-level instruction set architecture
- **Nsight Compute**: Use to understand actual hardware utilization
- **GPU Gems Series**: Deep dives into specific algorithms
- **Academic Papers**: MICRO, ISCA, ASPLOS conferences for cutting-edge research

---

*For practical programming guidance, see the main tutorial files.*

*For profiling these hardware features, see [09_profiling_debugging.md](09_profiling_debugging.md)*

