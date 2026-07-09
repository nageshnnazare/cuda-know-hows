# 11 — Matrix Multiplication (the canonical optimization journey)

> Part of **[CUDA Know-Hows](README.md)**. Prev: [10 — GPU architecture](10_gpu_architecture.md).
> Next: [12 — Atomics & synchronization](12_atomics_and_synchronization.md).
> Runnable code: [`examples/05_matrix_operations.cu`](examples/05_matrix_operations.cu).
>
> Goal: matrix multiply (GEMM) is *the* teaching example — it ties together
> coalescing (Ch. 05), shared-memory tiling (Ch. 07), and occupancy (Ch. 08),
> and shows the full arc from naive → tiled → library. By the end you'll
> understand why a tuned GEMM hits ~peak and yours won't (and why that's fine).

---

## 1. The problem and why it's a great teacher

`C = A · B` for N×N matrices: `C[i][j] = Σ_k A[i][k]·B[k][j]`. Each output
element is a dot product of a row of A and a column of B.

```
        A (N x N)         B (N x N)          C (N x N)
      ┌───────────┐     ┌───────────┐     ┌───────────┐
   i  │ ─────────▶│     │  │        │     │     Cij   │
      │  row i    │  ×  │  │col j   │  =  │      ●    │   Cij = row_i(A) · col_j(B)
      │           │     │  ▼        │     │           │
      └───────────┘     └───────────┘     └───────────┘

   FLOPs = 2*N^3 (a multiply + add per k, N^3 times).
   Naive data movement = O(N^3) too -> arithmetic intensity ~O(1) -> memory-bound.
   The whole game: REUSE each loaded element O(N) times -> raise intensity ->
   compute-bound (roofline, Ch. 00).
```

---

## 2. Version 0: naive kernel (one thread per output)

```cpp
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row*N + k] * B[k*N + col];   // each thread reads a full row & col
        C[row*N + col] = sum;
    }
}
```

```
   WHY IT'S SLOW:
     - each thread reads N elements of A and N of B from GLOBAL memory
     - adjacent threads (same row, col, col+1) BOTH re-read all of A's row and
       overlapping B columns -> the same data is fetched from DRAM again and again
     - A[row*N + k]: within a warp, `col` varies -> row fixed -> all lanes read the
       SAME A element (ok, broadcast) but B[k*N + col] IS coalesced.
     - total global traffic ~ 2*N^3 reads -> memory-bound, ~1-5% of peak.
```

---

## 3. Version 1: shared-memory tiling (the key win)

Split C into TILE×TILE tiles. Each block computes one output tile by streaming
matching tiles of A and B through **shared memory**, so each loaded element is
reused TILE times. This is cache blocking on the GPU (Ch. 07, and `cpp-hpc` M03).

```cpp
#define TILE 16
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N/TILE; ++t) {           // slide tiles across k
        As[threadIdx.y][threadIdx.x] = A[row*N + (t*TILE + threadIdx.x)];  // coalesced
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];  // coalesced
        __syncthreads();                          // all loads done

        for (int k = 0; k < TILE; ++k)            // reuse the tile from SHARED mem
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                          // before overwriting the tile
    }
    if (row < N && col < N) C[row*N + col] = sum;
}
```

```
   TILING: each block loads TILE x TILE sub-blocks of A and B ONCE, reuses TILE times.

     A tiles ─┐                     C tile computed by this block:
     ┌──┬──┬──┐   B tiles           ┌──┬──┬──┐
     │▓▓│  │  │   ┌──┬──┬──┐        │▒▒│  │  │  ▒ = this block's output tile
     ├──┼──┼──┤   │▓▓│  │  │   =    ├──┼──┼──┤
     │  │  │  │   ├──┼──┼──┤        │  │  │  │  For each k-tile ▓: load A-tile & B-tile
     └──┴──┴──┘   │  │  │  │        └──┴──┴──┘  to shared, accumulate. Global reads
                  └──┴──┴──┘                    drop from O(N^3) to O(N^3/TILE).

   Global memory traffic shrinks by ~TILE x -> arithmetic intensity rises ->
   compute-bound. Typically 3-10x faster than naive. Remember bank-conflict
   padding ([TILE][TILE+1]) for some access patterns (Ch. 07).
```

---

## 4. Version 2: register blocking (each thread computes multiple outputs)

The next level: each thread computes a small MICRO-tile (e.g. 4×4 outputs),
holding partial sums in **registers** and reusing shared-memory loads across them.
This raises arithmetic intensity further and is how fast GEMMs are structured.

```
   THREAD does 4x4 outputs, holding 16 accumulators in registers:
     load a strip of A and B tile into shared once
     each thread reuses those shared values across its 16 register accumulators
   -> fewer shared-memory loads per FLOP -> approaches compute peak.
   (This is what CUTLASS/cuBLAS do, plus double-buffering and Tensor Cores.)
```

This hand-tuning gets complex fast (double buffering, vectorized `float4` loads,
avoiding bank conflicts, warp tiling). Which is exactly why you should usually...

---

## 5. Version 3: just call cuBLAS (and why it wins)

```cpp
#include <cublas_v2.h>
cublasHandle_t h; cublasCreate(&h);
float alpha = 1.0f, beta = 0.0f;
// cuBLAS is column-major; compute C = alpha*A*B + beta*C. A common trick is to
// compute B*A to get row-major C, or set transposes accordingly.
cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, dB, N, dA, N, &beta, dC, N);
cublasDestroy(h);
```

```
   PERFORMANCE ARC (illustrative, large N, one GPU):
     naive              ~1-5%  of peak
     shared tiling      ~15-30%
     + register blocking~40-60%
     cuBLAS / CUTLASS   ~80-95%   <── decades of tuning + Tensor Cores

   cuBLAS uses: multi-level tiling (registers/shared/L2), vectorized loads,
   double-buffered async copies (Ch. 15), Tensor Cores (Ch. 21), and
   architecture-specific kernels selected at runtime. You won't beat it for
   standard GEMM — and you shouldn't try.
```

> **The lesson, not the code:** understand tiling and register blocking so you can
> (a) fuse GEMM with other ops when a library call would round-trip data through
> DRAM, and (b) reason about performance. For plain GEMM, call cuBLAS/CUTLASS
> (Chapter 20). For Tensor Core GEMM in mixed precision, see Chapter 21.

---

## 6. Bonus: matrix transpose (a coalescing lesson)

Transpose looks trivial but is a classic coalescing exercise: a naive transpose
has coalesced reads but **uncoalesced writes** (or vice versa). The fix is to
stage a tile in shared memory so both global read and write are coalesced.

```cpp
#define T 32
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ float tile[T][T+1];              // +1 padding avoids bank conflicts
    int x = blockIdx.x*T + threadIdx.x;
    int y = blockIdx.y*T + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y*N + x];  // coalesced read
    __syncthreads();
    x = blockIdx.y*T + threadIdx.x;             // transposed block coords
    y = blockIdx.x*T + threadIdx.y;
    if (x < N && y < N) out[y*N + x] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}
```

```
   NAIVE transpose: out[x*N+y] = in[y*N+x]  -> writes stride by N -> uncoalesced.
   SHARED-MEM transpose: read a tile coalesced, write it out coalesced from shared.
   The [T][T+1] padding removes the bank conflict the column read would cause.
```

Run all versions and compare on your GPU:

```bash
cd examples && make 05_matrix_operations && ./05_matrix_operations
```

---

## 7. Key takeaways

- GEMM is **memory-bound when naive** (each element read O(N) times) and becomes
  **compute-bound** once you reuse data — the roofline in action.
- **Shared-memory tiling** cuts global traffic by ~TILE× and is the essential
  first optimization (3–10×).
- **Register blocking** (each thread computes a micro-tile) pushes toward peak;
  it's how fast GEMMs are built (plus double-buffering, vectorized loads, Tensor
  Cores).
- **cuBLAS/CUTLASS reach ~80–95% of peak** — use them for standard GEMM; hand-roll
  only to **fuse** with surrounding ops or for shapes libraries don't cover.
- **Transpose** teaches coalescing: stage a padded tile in shared memory to
  coalesce both the read and the write.

**Next:** [12 — Atomics & synchronization →](12_atomics_and_synchronization.md)
