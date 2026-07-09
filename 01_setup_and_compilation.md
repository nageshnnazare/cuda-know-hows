# 01 — Setup & Compilation

> Part of **[CUDA Know-Hows](README.md)**. Prev: [00 — Introduction](00_introduction.md).
> Next: [02 — Your first kernel](02_first_kernel.md).
>
> Goal: get a working toolchain and *understand what `nvcc` actually does* — the
> host/device split, PTX vs SASS, architecture flags, and the lifecycle of a
> CUDA program. This is the plumbing every later chapter relies on.

---

## 1. What you need

```
   ┌────────────────────────────────────────────────────────────────────┐
   │ 1. An NVIDIA GPU        compute capability >= 7.0 recommended      │
   │                         (check: nvidia-smi --query-gpu=compute_cap │
   │                                 --format=csv)                      │
   │ 2. NVIDIA Driver        R550+ for CUDA 12.x, R580+ for CUDA 13.x   │
   │ 3. CUDA Toolkit         12.6+ (13.x recommended) — provides nvcc,  │
   │                         libraries (cuBLAS...), and headers         │
   │ 4. A host C++ compiler  GCC 7-14 / Clang 11+ / MSVC 2019+          │
   └────────────────────────────────────────────────────────────────────┘
```

Verify a working install:

```bash
nvidia-smi        # driver + GPU present, shows GPUs, driver & CUDA driver version
nvcc --version    # toolkit + compiler present, shows toolkit version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

```
   nvidia-smi shows the DRIVER's max supported CUDA version.
   nvcc --version shows the TOOLKIT version you compile with.
   The toolkit version must be <= what the driver supports (or use forward-compat).
```

Install on Linux (Ubuntu example) via NVIDIA's repo (preferred over distro
packages, which lag):

```bash
# Follow the exact commands from https://developer.nvidia.com/cuda-downloads
# for your OS/version, then set the environment:
echo 'export PATH=/usr/local/cuda/bin:$PATH'            >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

> **No GPU?** You can still learn: read every chapter, use
> [Google Colab](https://colab.research.google.com) (free T4 GPUs — write `.cu`
> in a cell and `!nvcc`), or [Compiler Explorer](https://godbolt.org) to inspect
> PTX/SASS without running. The conceptual chapters (10, 21) need no hardware.

---

## 2. `nvcc`: the CUDA compiler driver

`nvcc` is not a compiler — it's a **driver** that orchestrates several tools. It
splits your `.cu` file into **host code** (compiled by your normal C++ compiler)
and **device code** (compiled by NVIDIA's tools to GPU assembly), then stitches
the results into one executable.

```
   your_program.cu
        │
        ▼
   ┌─────────────────────────── nvcc ───────────────────────────┐
   │                                                            │
   │   split into host and device parts                         │
   │      │                                    │                │
   │      ▼ HOST code                          ▼ DEVICE code    │
   │   ┌──────────────┐                   ┌──────────────────┐  │
   │   │ gcc/clang/   │                   │ cicc  -> PTX     │  │
   │   │ msvc         │                   │ (virtual ISA)    │  │
   │   │ compiles CPU │                   │   │              │  │
   │   │ code + <<<>>>│                   │   ▼ ptxas        │  │
   │   │ launch stubs │                   │ SASS (real GPU   │  │
   │   └──────┬───────┘                   │ machine code)    │  │
   │          │                           └────────┬─────────┘  │
   │          │     fatbinary (PTX + SASS embedded)│            │
   │          └────────────────┬───────────────────┘            │
   │                           ▼                                │
   │                    linker -> a.out                         │
   └────────────────────────────────────────────────────────────┘
```

```
   TWO LEVELS OF DEVICE CODE — important to understand:
   PTX  = a VIRTUAL, forward-compatible assembly (an ISA the driver can JIT).
   SASS = the REAL machine code for a specific GPU architecture (sm_XX).

   The compiled binary is a "fatbinary" that can embed:
     - SASS for the exact architectures you targeted (runs immediately), AND/OR
     - PTX (which the driver JIT-compiles to SASS at runtime for NEWER GPUs).
   This is how one binary can run on future GPUs it wasn't built for.
```

Basic compile & run:

```bash
nvcc hello.cu -o hello        # compile
./hello                       # run
```

---

## 3. Architecture flags: `-arch`, `-gencode`, `-code`

You must tell `nvcc` which GPU(s) to generate code for. This is the flag people
get wrong most often (and why a binary throws "no kernel image is available").

```
   COMPUTE CAPABILITY (sm_XX) = the GPU's feature/ISA version. Each generation
   has one. Newer features (Tensor Cores, TMA, FP8) require newer sm_XX.

   generation      compute cap   flag           example GPUs
   ─────────────────────────────────────────────────────────────────────
   Volta           7.0           sm_70          V100, Titan V
   Turing          7.5           sm_75          RTX 20xx, T4
   Ampere          8.0 / 8.6     sm_80 / sm_86  A100 / RTX 30xx
   Ada Lovelace    8.9           sm_89          RTX 40xx, L40
   Hopper          9.0           sm_90(a)       H100, H200
   Blackwell DC    10.0          sm_100(a)      B200, GB200
   Blackwell cons. 12.0          sm_120         RTX 50xx
```

```bash
# Target ONE architecture (fast build, only runs on that arch family):
nvcc -arch=sm_80 prog.cu -o prog

# 'native' = detect and target the GPU in THIS machine (CUDA 12.something+):
nvcc -arch=native prog.cu -o prog

# Target MULTIPLE arches + embed PTX for forward compatibility (release builds):
nvcc -gencode arch=compute_80,code=sm_80 \    # SASS for Ampere
     -gencode arch=compute_90,code=sm_90 \    # SASS for Hopper
     -gencode arch=compute_90,code=compute_90 \  # PTX fallback for newer GPUs
     prog.cu -o prog
```

```
   -gencode arch=compute_XX,code=sm_YY  breaks down as:
       arch=compute_XX  -> the VIRTUAL arch (PTX) to compile the source to
       code=sm_YY       -> the REAL arch (SASS) to assemble for
   Include a code=compute_XX entry to also embed PTX -> JIT for future GPUs.

   Use the 'a' suffix (sm_90a, sm_100a) to enable ARCHITECTURE-SPECIFIC features
   (e.g. Hopper WGMMA/TMA, Blackwell tcgen05) that aren't forward-portable.
```

---

## 4. The flags you'll actually use

```
   -O3                    optimize device + host code
   -std=c++17 / c++20     C++ standard (c++20 needed for CUDA Tile, Ch. 21)
   -g -G                  debug: -g host symbols, -G DEVICE symbols (disables
                          device optimization! debug builds are much slower)
   -lineinfo              source line mapping for the profiler WITHOUT -G's
                          deoptimization — use this for profiling release builds
   --use_fast_math        faster, less precise math intrinsics (like CPU
                          -ffast-math; changes results — use deliberately)
   -Xcompiler "..."       pass flags to the HOST compiler (e.g. -Xcompiler -Wall)
   -Xptxas -v             print per-kernel register & shared-memory usage
                          (essential for occupancy tuning, Ch. 08)
   -maxrregcount=N        cap registers per thread (trade spills for occupancy)
   -rdc=true              relocatable device code: needed for dynamic parallelism
                          and cross-file device linking (Ch. 14)
   -lcublas -lcurand ...  link CUDA libraries (Ch. 20)
```

Example release + profiling build:

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo -Xptxas -v prog.cu -o prog
```

> **`-Xptxas -v` is your friend.** It prints, per kernel, `registers`,
> `smem`, `cmem` usage. Registers-per-thread and shared-memory-per-block are the
> two numbers that cap occupancy (Chapter 08). Read them every time you tune.

---

## 5. Inspecting the generated code (PTX & SASS)

You don't have to guess what the compiler produced — look at it.

```bash
# Emit PTX (virtual assembly):
nvcc -ptx prog.cu -o prog.ptx

# Disassemble the real SASS from a built binary:
cuobjdump -sass prog          # human-readable GPU machine code
nvdisasm prog.cubin           # alternative disassembler

# See which architectures a fatbinary contains:
cuobjdump prog | grep arch
```

[Compiler Explorer (godbolt.org)](https://godbolt.org) shows PTX/SASS live for
any CUDA version and arch — invaluable for confirming vectorized loads, checking
register pressure, or seeing whether a loop unrolled.

---

## 6. The CUDA program lifecycle

Every CUDA program (with explicit memory) follows the same shape. Keep this
picture in mind — Chapters 02 and 05 flesh out each step.

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                    CANONICAL CUDA PROGRAM FLOW                     │
   ├────────────────────────────────────────────────────────────────────┤
   │  1. allocate + init data on the HOST (CPU)                         │
   │  2. cudaMalloc device buffers on the GPU                           │
   │  3. cudaMemcpy  HOST -> DEVICE   (inputs across PCIe/NVLink)       │
   │  4. kernel<<<grid, block>>>(...) launch (ASYNC w.r.t. the host!)   │
   │  5. cudaMemcpy  DEVICE -> HOST   (results back)  [implicitly syncs]│
   │  6. cudaFree device buffers                                        │
   │  7. use results on the HOST                                        │
   └────────────────────────────────────────────────────────────────────┘

   With UNIFIED memory (cudaMallocManaged, Ch. 02/06) steps 2-3-5 collapse:
   the driver migrates pages on demand, so you allocate once and just launch.
```

```
   KEY GOTCHA: kernel launches are ASYNCHRONOUS. Step 4 returns to the CPU
   immediately, before the kernel finishes. You must synchronize (a following
   cudaMemcpy, or cudaDeviceSynchronize()) before trusting the results — and to
   see any errors the kernel raised (Ch. 02 error checking).
```

---

## 7. Build systems

For single files, `nvcc` on the command line (or the [`examples/Makefile`](examples/Makefile))
is fine. For projects, **CMake** has first-class CUDA support:

```cmake
cmake_minimum_required(VERSION 3.18)
project(myapp LANGUAGES CXX CUDA)          # CUDA is a first-class language
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 90)         # sm_80 + sm_90 (+ PTX automatically)
add_executable(myapp main.cu kernels.cu)
target_link_libraries(myapp CUDA::cublas)   # find_package(CUDAToolkit) targets
```

The [`examples/`](examples/) directory uses a Makefile; set `ARCH` to your GPU:

```bash
cd examples
make ARCH=sm_86 02_first_kernel     # build one example for your GPU
./02_first_kernel
```

---

## 8. Key takeaways

- `nvidia-smi` (driver) and `nvcc --version` (toolkit) verify your setup; the
  toolkit version must be supported by the driver.
- **`nvcc` is a driver** that splits `.cu` into host code (your C++ compiler) and
  device code (**PTX** virtual ISA → **SASS** real machine code), packed into a
  fatbinary.
- **Set `-arch`/`-gencode` correctly**: embed SASS for your target GPUs and PTX
  for forward-compat. `-arch=native` targets the local GPU; the `a` suffix
  unlocks arch-specific features.
- Build with `-O3 -std=c++17 -lineinfo -Xptxas -v`; use `-g -G` only for
  debugging (it deoptimizes device code).
- **Inspect** PTX/SASS with `cuobjdump`/godbolt; read register & shared-memory
  usage from `-Xptxas -v`.
- Internalize the **program lifecycle** and remember: **kernel launches are
  asynchronous** — synchronize before trusting results.

**Next:** [02 — Your first kernel →](02_first_kernel.md)
