# CUDA Know-Hows — Runnable Examples

All the runnable CUDA C++ for the guide lives here, built by one
[`Makefile`](Makefile). Each file is heavily commented and pairs with a chapter
in the [main guide](../README.md). Read the chapter, then read & run the code.

---

## Quick start

```bash
# 1. Check your setup
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 2. Set ARCH to YOUR GPU's compute capability, then build one example:
make ARCH=sm_86 02_first_kernel      # e.g. RTX 30xx = sm_86
./02_first_kernel

# 3. Or build everything / run everything / clean:
make ARCH=sm_86            # build all
make run                   # run all
make clean                 # remove binaries
make debug                 # build with -g -G (device debug)
make help                  # list targets
```

```
   ARCH values:  Volta sm_70 | Turing sm_75 | Ampere sm_80/86 | Ada sm_89
                 Hopper sm_90(a) | Blackwell sm_100(a)/sm_120
   Find yours:   nvidia-smi --query-gpu=compute_cap --format=csv
   The default ARCH in the Makefile is sm_80 — override on the command line.
```

Some examples link extra libraries (handled by the Makefile): cuBLAS
(`05`, `18`), cuRAND (`14`, `18`, `21`), and dynamic parallelism / device linking
(`08`, via `-rdc=true -lcudadevrt`).

---

## Example → chapter map

| Example | Demonstrates | Chapter |
|---------|--------------|---------|
| `02_first_kernel.cu` | `__global__`, launch, memcpy, error checking, vector add | [02](../02_first_kernel.md) |
| `03_memory_model.cu` | coalesced vs strided, constant memory, bandwidth | [05](../05_memory_model.md) |
| `04_thread_organization.cu` | 1D/2D/3D indexing, block-size effects | [03](../03_thread_hierarchy.md) / [04](../04_thread_indexing_patterns.md) |
| `05_matrix_operations.cu` | naive → tiled → cuBLAS GEMM, transpose | [11](../11_matrix_multiplication.md) |
| `06_shared_memory.cu` | reduction, scan, histogram, bank conflicts | [07](../07_shared_memory.md) |
| `07_streams_async.cu` | streams, async copy, overlap | [13](../13_streams_and_concurrency.md) |
| `08_advanced_topics.cu` | atomics, warp shuffle, dynamic parallelism | [12](../12_atomics_and_synchronization.md) / [14](../14_advanced_kernel_techniques.md) |
| `17_advanced_memory.cu` | async copy, vectorized loads, prefetch | [15](../15_advanced_memory_techniques.md) |
| `19_multi_gpu.cu` | device management, P2P | [17](../17_multi_gpu.md) |
| `22_modern_cuda.cu` | Tensor Cores / modern features | [21](../21_modern_cuda.md) |
| `gpu_locks_and_synchronization.cu` | atomics, locks, sync primitives in depth | [12](../12_atomics_and_synchronization.md) |
| `12_image_processing.cu` | convolution, filters, 2D tiling | [22](../22_applications.md) |
| `13_sorting_algorithms.cu` | bitonic / radix / merge sort | [22](../22_applications.md) |
| `14_scientific_computing.cu` | stencils, N-body, Monte Carlo | [22](../22_applications.md) |
| `16_graph_algorithms.cu` | BFS, connected components, PageRank | [22](../22_applications.md) |
| `18_ml_primitives.cu` | GEMM, softmax, normalization | [22](../22_applications.md) |
| `21_deep_learning.cu` | NN forward/backprop from scratch | [22](../22_applications.md) |

(`02_first_kernel_output.txt` is sample output for reference.)

---

## Notes

- These are **teaching** programs: clarity first, with the optimized version shown
  alongside the naive one so you can see the difference.
- Every kernel checks errors; run under `compute-sanitizer ./prog` to verify no
  out-of-bounds / races (see [Chapter 18](../18_profiling_and_debugging.md)).
- Performance numbers depend heavily on your GPU — the *relative* speedups between
  versions are the point, not the absolute times.
- No GPU handy? See [Chapter 01](../01_setup_and_compilation.md) for Colab /
  Compiler Explorer options.
