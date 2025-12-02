# GPU Architecture Internals: A Deep Dive
## Advanced Hardware Architecture Guide

This document provides an in-depth exploration of NVIDIA GPU architecture at the hardware level, intended for advanced users, performance engineers, and those seeking to understand the physical implementation details.

---

## Table of Contents

1. [GPU Die Architecture](#gpu-die-architecture)
2. [Streaming Multiprocessor (SM) Deep Dive](#streaming-multiprocessor-deep-dive)
3. [Execution Units and Pipelines](#execution-units-and-pipelines)
4. [Memory Subsystem Architecture](#memory-subsystem-architecture)
5. [Warp Scheduling and Execution](#warp-scheduling-and-execution)
6. [Cache Hierarchy Details](#cache-hierarchy-details)
7. [Memory Controllers and Interfaces](#memory-controllers-and-interfaces)
8. [Interconnect Architecture](#interconnect-architecture)
9. [Architecture Evolution](#architecture-evolution)
10. [Performance Characteristics](#performance-characteristics)

---

## GPU Die Architecture

### Complete GPU Chip Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         NVIDIA GPU DIE (Example: Ampere GA102)               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        GIGATHREAD ENGINE                             │    │
│  │  • Work Distribution Unit                                            │    │
│  │  • Global Scheduling                                                 │    │
│  │  • Thread Block Management                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                           │
│                                   ↓                                           │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐ │
│  │  GPC 0 │  GPC 1 │  GPC 2 │  GPC 3 │  GPC 4 │  GPC 5 │  GPC 6 │  GPC 7 │ │
│  │┌──────┐│┌──────┐│┌──────┐│┌──────┐│┌──────┐│┌──────┐│┌──────┐│┌──────┐│ │
│  ││ TPC  ││││ TPC  ││││ TPC  ││││ TPC  ││││ TPC  ││││ TPC  ││││ TPC  ││││ TPC  │││ │
│  ││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐││ │
│  │││SM 0││││││SM 2││││││SM 4││││││SM 6││││││SM 8││││││SM10││││││SM12││││││SM14│││ │
│  ││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘││ │
│  ││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐│││┌────┐││ │
│  │││SM 1││││││SM 3││││││SM 5││││││SM 7││││││SM 9││││││SM11││││││SM13││││││SM15│││ │
│  ││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘│││└────┘││ │
│  │└──────┘│└──────┘│└──────┘│└──────┘│└──────┘│└──────┘│└──────┘│└──────┘│ │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘ │
│         │        │        │        │        │        │        │        │      │
│  ┌──────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┐ │
│  │                              L2 CACHE                                   │ │
│  │         6 MB Partitioned across memory controllers                      │ │
│  │  [768KB] [768KB] [768KB] [768KB] [768KB] [768KB] [768KB] [768KB]      │ │
│  └──────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┘ │
│         │        │        │        │        │        │        │              │
│  ┌──────┴───┐┌──┴────┐┌──┴────┐┌──┴────┐┌──┴────┐┌──┴────┐┌──┴────┐      │
│  │  MEM     ││  MEM  ││  MEM  ││  MEM  ││  MEM  ││  MEM  ││  MEM  │      │
│  │  CTRL 0  ││ CTRL1 ││ CTRL2 ││ CTRL3 ││ CTRL4 ││ CTRL5 ││ CTRL6 │      │
│  │  64-bit  ││ 64-bit││ 64-bit││ 64-bit││ 64-bit││ 64-bit││ 64-bit│      │
│  └──────────┘└───────┘└───────┘└───────┘└───────┘└───────┘└───────┘      │
│       ↕          ↕         ↕         ↕         ↕         ↕         ↕        │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    GDDR6X MEMORY INTERFACE                            │ │
│  │                  448-bit bus width (7 × 64-bit)                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  KEY COMPONENTS:                                                              │
│  • GPC = Graphics Processing Cluster                                         │
│  • TPC = Texture Processing Cluster                                          │
│  • SM  = Streaming Multiprocessor                                            │
│  • MEM CTRL = Memory Controller                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical Organization

```
GPU Die
  │
  ├─ Gigathread Engine (Global Scheduler)
  │
  ├─ Graphics Processing Clusters (GPCs)
  │   │
  │   └─ Texture Processing Clusters (TPCs)
  │       │
  │       └─ Streaming Multiprocessors (SMs)
  │           │
  │           ├─ Processing Blocks (4 per SM)
  │           │   ├─ CUDA Cores (FP32/INT32)
  │           │   ├─ Tensor Cores
  │           │   └─ Special Function Units (SFU)
  │           │
  │           ├─ Warp Schedulers
  │           ├─ Register File
  │           ├─ Shared Memory / L1 Cache
  │           └─ Load/Store Units
  │
  ├─ L2 Cache (Shared across all SMs)
  │
  ├─ Memory Controllers
  │
  └─ Memory Interface (GDDR6/HBM)
```

---

## Streaming Multiprocessor (SM) Deep Dive

### SM Block Diagram (Ampere Architecture)

```
┌────────────────────────────────────────────────────────────────────────────┐
│              STREAMING MULTIPROCESSOR (SM) - Ampere GA10x                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                   WARP SCHEDULER & DISPATCH                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │    │
│  │  │  Scheduler 0 │  │  Scheduler 1 │  │  Scheduler 2 │            │    │
│  │  │  (Warp 0-15) │  │ (Warp 16-31) │  │ (Warp 32-47) │            │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │    │
│  └─────────┼──────────────────┼──────────────────┼────────────────────┘    │
│            │                  │                  │                          │
│            ↓                  ↓                  ↓                          │
│  ┌─────────────────────┬─────────────────────┬─────────────────────┐      │
│  │  PROCESSING BLOCK 0 │  PROCESSING BLOCK 1 │  PROCESSING BLOCK 2 │      │
│  ├─────────────────────┼─────────────────────┼─────────────────────┤      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │  FP32 Units   │  │  │  FP32 Units   │  │  │  FP32 Units   │  │      │
│  │  │  (16 cores)   │  │  │  (16 cores)   │  │  │  (16 cores)   │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │  INT32 Units  │  │  │  INT32 Units  │  │  │  INT32 Units  │  │      │
│  │  │  (16 cores)   │  │  │  (16 cores)   │  │  │  (16 cores)   │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │  FP64 Units   │  │  │  FP64 Units   │  │  │  FP64 Units   │  │      │
│  │  │  (1 core)     │  │  │  (1 core)     │  │  │  (1 core)     │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │ Tensor Core   │  │  │ Tensor Core   │  │  │ Tensor Core   │  │      │
│  │  │ (1 unit)      │  │  │ (1 unit)      │  │  │ (1 unit)      │  │      │
│  │  │ 4×4×4 MMA     │  │  │ 4×4×4 MMA     │  │  │ 4×4×4 MMA     │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │     SFU       │  │  │     SFU       │  │  │     SFU       │  │      │
│  │  │ (4 units)     │  │  │ (4 units)     │  │  │ (4 units)     │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  │  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐  │      │
│  │  │   LD/ST       │  │  │   LD/ST       │  │  │   LD/ST       │  │      │
│  │  │  (4 units)    │  │  │  (4 units)    │  │  │  (4 units)    │  │      │
│  │  └───────────────┘  │  └───────────────┘  │  └───────────────┘  │      │
│  │                     │                     │                     │      │
│  └─────────────────────┴─────────────────────┴─────────────────────┘      │
│                                   ↕                                         │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                      REGISTER FILE                                │     │
│  │          65,536 × 32-bit registers (256 KB total)                │     │
│  │  Dynamically allocated across active threads                     │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                   ↕                                         │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │              SHARED MEMORY / L1 CACHE (128 KB)                   │     │
│  │  ┌──────────────────────────────────────────────────────────┐   │     │
│  │  │  Configurable split:                                      │   │     │
│  │  │  • 100 KB Shared Memory + 28 KB L1 Cache                 │   │     │
│  │  │  • 68 KB Shared Memory + 60 KB L1 Cache                  │   │     │
│  │  │  • 36 KB Shared Memory + 92 KB L1 Cache                  │   │     │
│  │  └──────────────────────────────────────────────────────────┘   │     │
│  │                  32 banks × 4 bytes/clock                        │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                   ↕                                         │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                    TEXTURE / L1 CACHE                             │     │
│  │                      (32 KB per SM)                               │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                   ↕                                         │
│                            [To L2 Cache]                                    │
│                                                                             │
│  SPECIFICATIONS (per SM):                                                   │
│  • 128 FP32 CUDA Cores                                                     │
│  • 64 INT32 Cores                                                          │
│  • 4 FP64 Cores                                                            │
│  • 4 Tensor Cores (3rd gen)                                                │
│  • 4 Warp Schedulers                                                       │
│  • 48 Warps (max concurrent)                                               │
│  • 1,536 Threads (max)                                                     │
│  • 256 KB Register File                                                    │
│  • 128 KB Shared Memory/L1                                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### Processing Block Internal Structure

Each SM contains 4 processing blocks. Here's the detailed structure of one block:

```
┌──────────────────────────────────────────────────────────────┐
│              PROCESSING BLOCK (1 of 4 per SM)                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  FROM WARP SCHEDULER:                                         │
│  └─→ [Instruction Dispatch Queue] (2-way issue width)        │
│                       │                                       │
│                       ↓                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         FP32 EXECUTION UNITS (16 units)                │  │
│  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐   │  │
│  │  │FMA ││FMA ││FMA ││FMA ││FMA ││FMA ││FMA ││FMA │   │  │
│  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘   │  │
│  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐   │  │
│  │  │FMA ││FMA ││FMA ││FMA ││FMA ││FMA ││FMA ││FMA │   │  │
│  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘   │  │
│  │                                                        │  │
│  │  Each FMA: a×b + c (fused multiply-add)              │  │
│  │  Throughput: 16 FP32 ops/clock (32 with FMA)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ↕                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         INT32 EXECUTION UNITS (16 units)               │  │
│  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐   │  │
│  │  │ALU ││ALU ││ALU ││ALU ││ALU ││ALU ││ALU ││ALU │   │  │
│  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘   │  │
│  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐   │  │
│  │  │ALU ││ALU ││ALU ││ALU ││ALU ││ALU ││ALU ││ALU │   │  │
│  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘   │  │
│  │                                                        │  │
│  │  Operations: ADD, SUB, shift, logical, etc.          │  │
│  │  Concurrent execution with FP32                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ↕                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         TENSOR CORE (1 unit)                           │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  4×4×4 Matrix Multiply-Accumulate                │ │  │
│  │  │                                                   │ │  │
│  │  │  D = A × B + C                                   │ │  │
│  │  │                                                   │ │  │
│  │  │  Supported precisions:                           │ │  │
│  │  │  • FP16 / BF16  → FP32/FP16 accumulate          │ │  │
│  │  │  • TF32         → FP32 accumulate                │ │  │
│  │  │  • INT8 / INT4  → INT32 accumulate               │ │  │
│  │  │  • FP64         → FP64 accumulate (sparse)       │ │  │
│  │  │                                                   │ │  │
│  │  │  Throughput: 256 FP16 ops/clock                  │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ↕                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │    SPECIAL FUNCTION UNITS (SFU) (4 units)              │  │
│  │  ┌────┐┌────┐┌────┐┌────┐                            │  │
│  │  │SFU ││SFU ││SFU ││SFU │                            │  │
│  │  └────┘└────┘└────┘└────┘                            │  │
│  │                                                        │  │
│  │  Operations:                                           │  │
│  │  • Transcendental functions (sin, cos, log, exp)     │  │
│  │  • Square root, reciprocal                            │  │
│  │  • Interpolation                                      │  │
│  │                                                        │  │
│  │  1/4 throughput of FP32 cores                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ↕                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │       LOAD/STORE UNITS (4 units)                       │  │
│  │  ┌────┐┌────┐┌────┐┌────┐                            │  │
│  │  │LD/ST││LD/ST││LD/ST││LD/ST│                         │  │
│  │  └────┘└────┘└────┘└────┘                            │  │
│  │                                                        │  │
│  │  • Memory address calculation                         │  │
│  │  • Shared memory access                               │  │
│  │  • Global memory access                               │  │
│  │  • Texture/surface memory access                      │  │
│  │  • Atomic operations                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ↕                                       │
│              [Register File Bank]                             │
│              [Shared Memory Bank]                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Execution Units and Pipelines

### FP32 CUDA Core Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│               FP32 CUDA CORE PIPELINE STAGES                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Stage 1: FETCH                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  • Instruction fetch from instruction cache            │      │
│  │  • PC (Program Counter) update                         │      │
│  └────────────────────────────────────────────────────────┘      │
│                         ↓                                         │
│  Stage 2: DECODE                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  • Instruction decode                                  │      │
│  │  • Register address decode                             │      │
│  │  • Operand collection                                  │      │
│  └────────────────────────────────────────────────────────┘      │
│                         ↓                                         │
│  Stage 3: READ REGISTERS                                          │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  • Register file access                                │      │
│  │  • Bank conflict resolution                            │      │
│  │  • Operand forwarding check                            │      │
│  └────────────────────────────────────────────────────────┘      │
│                         ↓                                         │
│  Stage 4: EXECUTE (FMA - Fused Multiply-Add)                     │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                                                         │      │
│  │  Input: A (multiplicand), B (multiplier), C (addend)  │      │
│  │                                                         │      │
│  │  ┌─────────────────────┐                               │      │
│  │  │  Mantissa Multiply  │                               │      │
│  │  │  (24-bit × 24-bit)  │                               │      │
│  │  └──────────┬──────────┘                               │      │
│  │             ↓                                            │      │
│  │  ┌─────────────────────┐                               │      │
│  │  │  Exponent Add       │                               │      │
│  │  │  & Alignment        │                               │      │
│  │  └──────────┬──────────┘                               │      │
│  │             ↓                                            │      │
│  │  ┌─────────────────────┐                               │      │
│  │  │  Mantissa Add (48b) │                               │      │
│  │  │  + C mantissa       │                               │      │
│  │  └──────────┬──────────┘                               │      │
│  │             ↓                                            │      │
│  │  ┌─────────────────────┐                               │      │
│  │  │  Normalize & Round  │                               │      │
│  │  └──────────┬──────────┘                               │      │
│  │             ↓                                            │      │
│  │         Result (32-bit)                                 │      │
│  │                                                         │      │
│  │  Latency: ~4 cycles                                    │      │
│  │  Throughput: 1 operation/cycle (pipelined)             │      │
│  └────────────────────────────────────────────────────────┘      │
│                         ↓                                         │
│  Stage 5: WRITEBACK                                               │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  • Result written to register file                     │      │
│  │  • Dependency resolution                               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  CHARACTERISTICS:                                                 │
│  • Pipeline depth: ~5 stages                                     │
│  • Issue latency: 4 cycles                                       │
│  • Throughput: 2 ops/cycle (FMA = multiply + add)               │
│  • Full IEEE 754-2008 compliance                                 │
│  • Supports: FP32, FP16 (2× rate with Tensor cores)             │
└──────────────────────────────────────────────────────────────────┘
```

### Tensor Core Operation

```
┌──────────────────────────────────────────────────────────────────┐
│              TENSOR CORE MATRIX OPERATION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  D = A × B + C    (4×4 × 4×4 + 4×4 matrix operation)            │
│                                                                   │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐                  │
│  │ A Matrix│      │ B Matrix│      │ C Matrix│                  │
│  │  (4×4)  │      │  (4×4)  │      │  (4×4)  │                  │
│  │┌──┬──┐  │      │┌──┬──┐  │      │┌──┬──┐  │                  │
│  ││a0│a1││  │  ×  ││b0│b1││  │  +  ││c0│c1││  │                  │
│  │├──┼──┤  │      │├──┼──┤  │      │├──┼──┤  │                  │
│  ││a2│a3││  │      ││b2│b3││  │      ││c2│c3││  │                  │
│  │└──┴──┘  │      │└──┴──┘  │      │└──┴──┘  │                  │
│  └────┬────┘      └────┬────┘      └────┬────┘                  │
│       │                │                 │                        │
│       └────────────────┼─────────────────┘                        │
│                        ↓                                          │
│  ┌──────────────────────────────────────────────────────┐        │
│  │         TENSOR CORE COMPUTE UNIT                     │        │
│  │                                                       │        │
│  │  Step 1: Parallel Multiply (16 muls simultaneously)  │        │
│  │  ┌─────────────────────────────────────────────┐    │        │
│  │  │  Row 0: a0×b0, a1×b2, a0×b1, a1×b3          │    │        │
│  │  │  Row 1: a2×b0, a3×b2, a2×b1, a3×b3          │    │        │
│  │  │  Row 2: (... 8 more multiplications)        │    │        │
│  │  │  Row 3: (... 8 more multiplications)        │    │        │
│  │  └─────────────────────────────────────────────┘    │        │
│  │                                                       │        │
│  │  Step 2: Reduction Tree (accumulate products)        │        │
│  │  ┌─────────────────────────────────────────────┐    │        │
│  │  │  Stage 1: 16 products → 8 sums               │    │        │
│  │  │  Stage 2:  8 sums → 4 sums                   │    │        │
│  │  │  Stage 3:  4 sums → 4 elements (per row)     │    │        │
│  │  └─────────────────────────────────────────────┘    │        │
│  │                                                       │        │
│  │  Step 3: Add C matrix (accumulator)                  │        │
│  │  ┌─────────────────────────────────────────────┐    │        │
│  │  │  d[i][j] = mul_accum[i][j] + c[i][j]        │    │        │
│  │  └─────────────────────────────────────────────┘    │        │
│  │                                                       │        │
│  │  Latency: ~8 cycles                                  │        │
│  │  Operations: 64 FMA = 128 operations                 │        │
│  │  Throughput: 128 ops / 8 cycles = 16 ops/cycle      │        │
│  └──────────────────────────────────────────────────────┘        │
│                        ↓                                          │
│  ┌─────────┐                                                      │
│  │ D Matrix│  Result (4×4)                                       │
│  │  (4×4)  │                                                      │
│  │┌──┬──┐  │                                                      │
│  ││d0│d1││  │                                                      │
│  │├──┼──┤  │                                                      │
│  ││d2│d3││  │                                                      │
│  │└──┴──┘  │                                                      │
│  └─────────┘                                                      │
│                                                                   │
│  PERFORMANCE:                                                     │
│  • Per Tensor Core: 256 FP16 ops/clock (with FMA)               │
│  • Per SM (4 TCs): 1024 FP16 ops/clock                           │
│  • 16× faster than FP32 cores for same operation                 │
│  • Supports: FP16, BF16, TF32, INT8, INT4, FP64 (sparse)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Memory Subsystem Architecture

### Complete Memory Hierarchy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY HIERARCHY ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PER-THREAD LEVEL:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  REGISTERS (Thread-Private)                                         │    │
│  │  • 32-bit registers                                                  │    │
│  │  • Up to 255 registers per thread                                   │    │
│  │  • Access latency: 1 cycle                                          │    │
│  │  • Bandwidth: Unlimited (local to thread)                           │    │
│  │  • Spills to local memory if exceeded                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  PER-SM LEVEL:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  REGISTER FILE (Per SM)                                             │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  256 KB (65,536 × 32-bit registers)                        │    │    │
│  │  │  ┌──────────┬──────────┬──────────┬──────────┐             │    │    │
│  │  │  │  Bank 0  │  Bank 1  │  Bank 2  │  Bank 3  │             │    │    │
│  │  │  │  (64 KB) │  (64 KB) │  (64 KB) │  (64 KB) │             │    │    │
│  │  │  └──────────┴──────────┴──────────┴──────────┘             │    │    │
│  │  │  Dynamically partitioned across warps                      │    │    │
│  │  │  Access: 4 banks/cycle × 32-bit = 128 bytes/cycle         │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SHARED MEMORY / L1 CACHE (Unified, Per SM)                        │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Total: 128 KB (Ampere)                                    │    │    │
│  │  │  ┌──────────────────────────────────────────────────┐     │    │    │
│  │  │  │  32 Memory Banks (4-byte wide each)              │     │    │    │
│  │  │  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐     │     │    │    │
│  │  │  │  │B0 │B1 │B2 │B3 │...│...│...│...│B30│B31│     │     │    │    │
│  │  │  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘     │     │    │    │
│  │  │  │  Each bank: 4 bytes/cycle                        │     │    │    │
│  │  │  └──────────────────────────────────────────────────┘     │    │    │
│  │  │                                                            │    │    │
│  │  │  Configurable split (Ampere):                            │    │    │
│  │  │  • Option 1: 100 KB Shared + 28 KB L1                    │    │    │
│  │  │  • Option 2:  68 KB Shared + 60 KB L1                    │    │    │
│  │  │  • Option 3:  36 KB Shared + 92 KB L1                    │    │    │
│  │  │                                                            │    │    │
│  │  │  Access latency: ~20 cycles (shared), ~30 cycles (L1)    │    │    │
│  │  │  Bandwidth: 128 bytes/cycle (all banks)                   │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TEXTURE CACHE / READ-ONLY CACHE (Per SM)                          │    │
│  │  • Size: 32 KB per SM                                               │    │
│  │  • Optimized for 2D spatial locality                               │    │
│  │  • Filtering and interpolation hardware                            │    │
│  │  • Latency: ~100 cycles                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  CHIP-WIDE LEVEL:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  L2 CACHE (Shared across all SMs)                                  │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Size: 6 MB (Ampere GA102), partitioned                   │    │    │
│  │  │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐      │    │    │
│  │  │  │768KB │768KB │768KB │768KB │768KB │768KB │768KB │      │    │    │
│  │  │  │Slice │Slice │Slice │Slice │Slice │Slice │Slice │      │    │    │
│  │  │  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘      │    │    │
│  │  │     │      │      │      │      │      │      │           │    │    │
│  │  │    MC0    MC1    MC2    MC3    MC4    MC5    MC6         │    │    │
│  │  │                                                            │    │    │
│  │  │  • Line size: 128 bytes                                  │    │    │
│  │  │  • Associativity: 16-way set associative                 │    │    │
│  │  │  • Latency: ~200 cycles                                   │    │    │
│  │  │  • Bandwidth: ~1500 GB/s internal                        │    │    │
│  │  │  • Atomic operations support                             │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MEMORY CONTROLLERS (7 × 64-bit = 448-bit bus)                     │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │  Each Controller:                                        │      │    │
│  │  │  • 64-bit interface width                                │      │    │
│  │  │  • Supports GDDR6X/GDDR6/HBM2                           │      │    │
│  │  │  • ECC support (optional)                                │      │    │
│  │  │  • Compression/Decompression engine                      │      │    │
│  │  │  • Outstanding request buffers                           │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   ↕                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GLOBAL MEMORY (GDDR6X/HBM)                                        │    │
│  │  • Capacity: 10-24 GB typical                                       │    │
│  │  • Bandwidth: 760 GB/s (GDDR6X), 1.5+ TB/s (HBM2e)                │    │
│  │  • Latency: ~400 cycles                                            │    │
│  │  • ECC protected (optional)                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  PERFORMANCE SUMMARY:                                                         │
│  ┌──────────────────┬────────────┬──────────────┬─────────────────────┐    │
│  │ Memory Type      │ Latency    │ Bandwidth    │ Size                │    │
│  ├──────────────────┼────────────┼──────────────┼─────────────────────┤    │
│  │ Registers        │  1 cycle   │ Unlimited    │ 256 KB/SM           │    │
│  │ Shared Memory    │ 20 cycles  │ 128 B/cycle  │ 128 KB/SM           │    │
│  │ L1 Cache         │ 30 cycles  │ 128 B/cycle  │ 28-92 KB/SM         │    │
│  │ Texture Cache    │ 100 cycles │ Variable     │ 32 KB/SM            │    │
│  │ L2 Cache         │ 200 cycles │ 1500 GB/s    │ 6 MB (chip-wide)    │    │
│  │ Global Memory    │ 400 cycles │ 760 GB/s     │ 10-24 GB            │    │
│  └──────────────────┴────────────┴──────────────┴─────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Shared Memory Bank Structure

```
┌──────────────────────────────────────────────────────────────────┐
│            SHARED MEMORY BANKING ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  32 BANKS × 4 BYTES = 128 bytes/clock access                    │
│                                                                   │
│  Address Layout:                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Byte Address: [.... address bits ....][bank][offset]  │      │
│  │                                         └─5b─┘└─2b──┘  │      │
│  │                                                         │      │
│  │  Bank ID = (address >> 2) & 0x1F  (bits 6:2)          │      │
│  │  Offset  = address & 0x3            (bits 1:0)          │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  Physical Layout (128 KB total):                                  │
│  ┌──────┬──────┬──────┬──────┬─────┬──────┬──────┬──────┐       │
│  │Bank 0│Bank 1│Bank 2│Bank 3│ ... │Bank30│Bank31│      │       │
│  │ 4KB  │ 4KB  │ 4KB  │ 4KB  │     │ 4KB  │ 4KB  │      │       │
│  ├──────┼──────┼──────┼──────┼─────┼──────┼──────┼──────┤       │
│  │Word 0│Word 0│Word 0│Word 0│ ... │Word 0│Word 0│      │       │
│  │Word32│Word32│Word32│Word32│     │Word32│Word32│      │       │
│  │Word64│Word64│Word64│Word64│     │Word64│Word64│      │       │
│  │  ...  │  ...  │  ...  │  ...  │     │  ...  │  ...  │      │       │
│  └──────┴──────┴──────┴──────┴─────┴──────┴──────┴──────┘       │
│                                                                   │
│  ACCESS PATTERNS:                                                 │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  NO CONFLICT (Ideal - All threads different banks):    │      │
│  │  ┌──────────────────────────────────────────────┐      │      │
│  │  │  T0 → Bank 0                                 │      │      │
│  │  │  T1 → Bank 1                                 │      │      │
│  │  │  T2 → Bank 2                                 │      │      │
│  │  │  ...                                         │      │      │
│  │  │  T31 → Bank 31                               │      │      │
│  │  │  Result: 1 cycle, 128 bytes transferred     │      │      │
│  │  └──────────────────────────────────────────────┘      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  2-WAY BANK CONFLICT:                                  │      │
│  │  ┌──────────────────────────────────────────────┐      │      │
│  │  │  T0, T1 → Bank 0  (conflict!)                │      │      │
│  │  │  T2, T3 → Bank 1  (conflict!)                │      │      │
│  │  │  ...                                         │      │      │
│  │  │  Result: 2 serialized accesses, 2 cycles    │      │      │
│  │  └──────────────────────────────────────────────┘      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  BROADCAST (Special Case - All same address):          │      │
│  │  ┌──────────────────────────────────────────────┐      │      │
│  │  │  T0, T1, T2, ... T31 → Bank 0, Address X    │      │      │
│  │  │  Result: 1 cycle (broadcast optimized)      │      │      │
│  │  └──────────────────────────────────────────────┘      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  PADDING TO AVOID CONFLICTS:                                      │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Without padding: float array[32][32]                  │      │
│  │  └─> array[tid][0] all map to Bank 0 (32-way conflict)│      │
│  │                                                         │      │
│  │  With padding: float array[32][33]                     │      │
│  │  └─> array[tid][0] maps to different banks (no conflict)│     │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Warp Scheduling and Execution

### Warp Scheduler Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    WARP SCHEDULER ARCHITECTURE (Per SM)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      WARP SCHEDULER UNIT                            │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │  Active Warp Pool (48 warps per SM)                      │      │    │
│  │  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐  │      │    │
│  │  │  │ W0 │ W1 │ W2 │ W3 │ W4 │ W5 │ W6 │ W7 │ ... │W47 │  │      │    │
│  │  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘  │      │    │
│  │  │                                                         │      │    │
│  │  │  Each warp: 32 threads                                 │      │    │
│  │  │  Per-warp state:                                       │      │    │
│  │  │  • Program Counter (PC)                                │      │    │
│  │  │  • Active mask (32-bit, 1 per thread)                 │      │    │
│  │  │  • Register allocation                                 │      │    │
│  │  │  • Execution state                                     │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  │                              ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │           WARP SELECTION LOGIC                          │      │    │
│  │  │  ┌────────────────────────────────────────────────┐    │      │    │
│  │  │  │  Priority Scheduling Algorithm:                │    │      │    │
│  │  │  │  1. Check warp eligibility:                   │    │      │    │
│  │  │  │     • Ready instruction in I-cache            │    │      │    │
│  │  │  │     • No data dependencies (scoreboarding)    │    │      │    │
│  │  │  │     • Execution unit available                │    │      │    │
│  │  │  │     • No memory stall                         │    │      │    │
│  │  │  │                                                │    │      │    │
│  │  │  │  2. Select highest priority ready warp        │    │      │    │
│  │  │  │     • Round-robin among ready warps           │    │      │    │
│  │  │  │     • Oldest instruction first                │    │      │    │
│  │  │  │     • Load balancing                          │    │      │    │
│  │  │  └────────────────────────────────────────────────┘    │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  │                              ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │           INSTRUCTION FETCH & DECODE                    │      │    │
│  │  │  ┌────────────────────────────────────────────────┐    │      │    │
│  │  │  │  • Fetch from instruction cache                │    │      │    │
│  │  │  │  • Decode instruction                          │    │      │    │
│  │  │  │  • Determine execution unit needed             │    │      │    │
│  │  │  │  • Extract operands & registers                │    │      │    │
│  │  │  └────────────────────────────────────────────────┘    │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  │                              ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │           SCOREBOARD (Dependency Tracking)              │      │    │
│  │  │  ┌────────────────────────────────────────────────┐    │      │    │
│  │  │  │  Tracks:                                       │    │      │    │
│  │  │  │  • Register read/write dependencies            │    │      │    │
│  │  │  │  • Memory operation status                     │    │      │    │
│  │  │  │  • Execution unit busy status                  │    │      │    │
│  │  │  │  • Outstanding memory requests                 │    │      │    │
│  │  │  │                                                │    │      │    │
│  │  │  │  Prevents:                                     │    │      │    │
│  │  │  │  • Read-after-write (RAW) hazards              │    │      │    │
│  │  │  │  • Write-after-write (WAW) hazards             │    │      │    │
│  │  │  │  • Resource conflicts                          │    │      │    │
│  │  │  └────────────────────────────────────────────────┘    │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  │                              ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐      │    │
│  │  │           DISPATCH TO EXECUTION UNITS                   │      │    │
│  │  │  ┌─────────┬─────────┬─────────┬─────────┬────────┐    │      │    │
│  │  │  │  FP32   │  INT32  │ TENSOR  │   SFU   │ LD/ST  │    │      │    │
│  │  │  │  PIPES  │  PIPES  │  CORE   │  PIPES  │ PIPES  │    │      │    │
│  │  │  └─────────┴─────────┴─────────┴─────────┴────────┘    │      │    │
│  │  └──────────────────────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              EXECUTION PIPELINE TIMELINE (4 schedulers)             │    │
│  │                                                                      │    │
│  │  Cycle:  0    1    2    3    4    5    6    7    8    9    10      │    │
│  │          │    │    │    │    │    │    │    │    │    │    │       │    │
│  │  Sched0: │W0──│W1──│W2──│W3──│W0──│W1──│W2──│W3──│W0──│W1──│      │    │
│  │  Sched1: │W4──│W5──│W6──│W7──│W4──│W5──│W6──│W7──│W4──│W5──│      │    │
│  │  Sched2: │W8──│W9──│W10─│W11─│W8──│W9──│W10─│W11─│W8──│W9──│      │    │
│  │  Sched3: │W12─│W13─│W14─│W15─│W12─│W13─│W14─│W15─│W12─│W13─│      │    │
│  │                                                                      │    │
│  │  Each scheduler can issue 1 instruction per cycle                   │    │
│  │  Up to 4 warps execute simultaneously (1 per scheduler)             │    │
│  │  Different warps hide latency through interleaving                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  LATENCY HIDING THROUGH MULTITHREADING:                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Instruction Latency Examples:                                      │    │
│  │  • FP32 arithmetic: 4 cycles                                        │    │
│  │  • Shared memory load: 20 cycles                                    │    │
│  │  • Global memory load: 400+ cycles                                  │    │
│  │                                                                      │    │
│  │  With 48 active warps:                                              │    │
│  │  • While W0 waits for memory (400 cycles)                           │    │
│  │  • Scheduler can execute W1, W2, W3, ..., W47                       │    │
│  │  • By the time we cycle back to W0, data is ready!                  │    │
│  │                                                                      │    │
│  │  Required warps to hide latency = Latency / Pipeline_depth          │    │
│  │  For 400-cycle latency: 400 / 4 = 100 warps needed (ideal)          │    │
│  │  Actual: 48 warps × 4 schedulers = 192 potential instructions       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Branch Divergence Handling

```
┌──────────────────────────────────────────────────────────────────┐
│                BRANCH DIVERGENCE HANDLING                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Code Example:                                                    │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  __global__ void divergent_kernel(int *data, int n) {  │      │
│  │      int tid = threadIdx.x;                           │      │
│  │      if (tid % 2 == 0) {           // DIVERGENCE!     │      │
│  │          data[tid] = compute_A();  // Branch A        │      │
│  │      } else {                                          │      │
│  │          data[tid] = compute_B();  // Branch B        │      │
│  │      }                                                 │      │
│  │  }                                                     │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  EXECUTION WITH DIVERGENCE:                                       │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                                                         │      │
│  │  Warp (32 threads): T0 T1 T2 T3 ... T30 T31           │      │
│  │                                                         │      │
│  │  Step 1: Evaluate condition                            │      │
│  │  ┌────────────────────────────────────────────────┐   │      │
│  │  │ Active Mask: 10101010...1010 (even threads)   │   │      │
│  │  │ Result: 16 threads take Branch A              │   │      │
│  │  │         16 threads take Branch B              │   │      │
│  │  └────────────────────────────────────────────────┘   │      │
│  │                                                         │      │
│  │  Step 2: Execute Branch A (even threads)               │      │
│  │  ┌────────────────────────────────────────────────┐   │      │
│  │  │ Active: T0  T2  T4  ... T28  T30              │   │      │
│  │  │ Masked: T1  T3  T5  ... T29  T31  (idle)      │   │      │
│  │  │                                                 │   │      │
│  │  │ Execute: compute_A()                           │   │      │
│  │  │ Warp Efficiency: 50% (16/32 threads active)   │   │      │
│  │  └────────────────────────────────────────────────┘   │      │
│  │                                                         │      │
│  │  Step 3: Execute Branch B (odd threads)                │      │
│  │  ┌────────────────────────────────────────────────┐   │      │
│  │  │ Active: T1  T3  T5  ... T29  T31              │   │      │
│  │  │ Masked: T0  T2  T4  ... T28  T30  (idle)      │   │      │
│  │  │                                                 │   │      │
│  │  │ Execute: compute_B()                           │   │      │
│  │  │ Warp Efficiency: 50% (16/32 threads active)   │   │      │
│  │  └────────────────────────────────────────────────┘   │      │
│  │                                                         │      │
│  │  Total Time = Time(Branch_A) + Time(Branch_B)         │      │
│  │  Average Efficiency: 50%                               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  DIVERGENCE STACK:                                                │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Hardware maintains a stack of execution paths:        │      │
│  │                                                         │      │
│  │  ┌───────────────────┐                                 │      │
│  │  │  Reconverge PC    │  ← Top                          │      │
│  │  ├───────────────────┤                                 │      │
│  │  │  Branch B mask    │                                 │      │
│  │  ├───────────────────┤                                 │      │
│  │  │  Branch A mask    │  ← Current                      │      │
│  │  ├───────────────────┤                                 │      │
│  │  │  Previous state   │                                 │      │
│  │  └───────────────────┘                                 │      │
│  │                                                         │      │
│  │  1. Push reconvergence point                           │      │
│  │  2. Execute first path with mask                       │      │
│  │  3. Pop, execute second path                           │      │
│  │  4. Reconverge all threads                             │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  OPTIMIZATION STRATEGIES:                                         │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  1. MINIMIZE DIVERGENCE:                               │      │
│  │     • Organize data so threads in same warp take       │      │
│  │       same path                                        │      │
│  │     • Use __ballot_sync() to handle divergence         │      │
│  │       explicitly                                        │      │
│  │                                                         │      │
│  │  2. PREDICATION (Compiler Optimization):               │      │
│  │     • Convert branches to conditional assignment       │      │
│  │     • result = condition ? val_a : val_b;              │      │
│  │     • No divergence, but both paths may execute        │      │
│  │                                                         │      │
│  │  3. WARP-LEVEL PRIMITIVES:                             │      │
│  │     • Use __any_sync(), __all_sync()                   │      │
│  │     • Detect uniform conditions                         │      │
│  │     • Early exit when possible                          │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

This is part 1 of the advanced GPU architecture internals guide. Would you like me to continue with the remaining sections covering cache hierarchy, memory controllers, interconnects, architecture evolution across generations, and detailed performance characteristics?
