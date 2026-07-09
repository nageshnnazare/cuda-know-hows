# 13 — Streams & Concurrency

> Part of **[CUDA Know-Hows](README.md)**. Prev: [12 — Atomics & synchronization](12_atomics_and_synchronization.md).
> Next: [14 — Advanced kernel techniques](14_advanced_kernel_techniques.md).
> Runnable code: [`examples/07_streams_async.cu`](examples/07_streams_async.cu).
>
> Goal: hide the PCIe transfer bottleneck and keep the GPU busy by overlapping
> copies with compute (and compute with compute). You'll learn streams, events,
> asynchronous copies, pinned memory, and the multi-stream pipelining pattern.

---

## 1. The problem: serial copy → compute → copy wastes the GPU

By default everything runs in the **default stream**, serialized: copy inputs up,
run the kernel, copy results down — one after another. During each copy the
compute units sit idle; during compute the copy engines sit idle.

```
   SERIAL (default stream), one big batch:
     timeline ─▶  [==== H2D copy ====][==== kernel ====][==== D2H copy ====]
                   copy engine busy    SMs busy          copy engine busy
                   SMs IDLE            copy engines IDLE  SMs IDLE
   The GPU is only ~1/3 utilized. PCIe (~16-32 GB/s) is slow vs VRAM (TB/s).
```

The fix: split the work into chunks and run them in **streams** so the copy of
one chunk overlaps the compute of another.

---

## 2. What a stream is

A **stream** is an ordered queue of GPU operations. Operations *within* a stream
run in order; operations in *different* streams may run **concurrently** (if the
hardware has the resources: separate copy engines + SMs).

```
   Stream = an in-order queue. Different streams = independent queues.

   default stream:  op1 -> op2 -> op3            (serialized)

   stream A:  H2D(a) -> kernel(a) -> D2H(a)  ┐
   stream B:  H2D(b) -> kernel(b) -> D2H(b)  ├─ these can OVERLAP each other
   stream C:  H2D(c) -> kernel(c) -> D2H(c)  ┘   (copy of one, compute of another)
```

```cpp
cudaStream_t s;
cudaStreamCreate(&s);
kernel<<<grid, block, 0, s>>>(...);          // 4th launch arg = the stream
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);  // async, in stream s
cudaStreamSynchronize(s);                     // wait for THIS stream only
cudaStreamDestroy(s);
```

---

## 3. Pinned (page-locked) memory is required for overlap

Asynchronous copies (`cudaMemcpyAsync`) only truly overlap when the **host**
buffer is *pinned* (page-locked). Pageable memory forces the driver to stage
through a pinned bounce buffer, serializing the copy.

```cpp
float* hBuf;
cudaMallocHost(&hBuf, bytes);        // pinned host memory (or cudaHostAlloc)
// ... use hBuf in cudaMemcpyAsync ...
cudaFreeHost(hBuf);
```

```
   pageable host memory + cudaMemcpyAsync -> copy is effectively SYNC (no overlap)
   pinned host memory   + cudaMemcpyAsync -> true async DMA -> overlap works
   BUT pinned memory is a scarce OS resource — pin only transfer buffers, not
   everything, or you'll degrade the whole system.
```

---

## 4. The pipelining pattern (overlap copy with compute)

Split the data into chunks; issue each chunk's H2D, kernel, and D2H into its own
stream. The copy engines and SMs then work on different chunks simultaneously.

```cpp
const int nStreams = 4, chunk = N / nStreams;
cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; ++i) cudaStreamCreate(&streams[i]);

for (int i = 0; i < nStreams; ++i) {
    int off = i * chunk;
    cudaMemcpyAsync(dA+off, hA+off, chunk*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    kernel<<<gridChunk, block, 0, streams[i]>>>(dA+off, dC+off, chunk);
    cudaMemcpyAsync(hC+off, dC+off, chunk*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
}
for (int i = 0; i < nStreams; ++i) cudaStreamSynchronize(streams[i]);
```

```
   OVERLAPPED PIPELINE (4 streams):

   copy engine H2D:  [H2D0][H2D1][H2D2][H2D3]
   SMs (compute):          [ K0 ][ K1 ][ K2 ][ K3 ]
   copy engine D2H:               [D2H0][D2H1][D2H2][D2H3]
                     └── while chunk 0 computes, chunk 1 copies in, etc. ──┘

   Total time drops from (copy+compute+copy) toward max(copy, compute).
   Real speedup often ~2-3x for transfer-heavy workloads.
```

```
   TIP: modern GPUs have separate H2D and D2H copy engines, so H2D of one chunk,
   compute of another, and D2H of a third can all run at once. The "depth-first"
   issue order above (per-stream H2D->K->D2H) enables this; a "breadth-first"
   order (all H2D, then all K, then all D2H) also works and is sometimes clearer.
```

---

## 5. Events: timing and cross-stream dependencies

**Events** are markers you record in a stream. Use them to time GPU work
precisely (Ch. 02) and to make one stream wait on another.

```cpp
cudaEvent_t e; cudaEventCreate(&e);

// Timing:
cudaEventRecord(start, s); kernel<<<...,s>>>(); cudaEventRecord(stop, s);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);

// Cross-stream dependency: make stream B wait for an event in stream A
cudaEventRecord(e, streamA);
cudaStreamWaitEvent(streamB, e, 0);    // B's next ops wait until e completes in A
```

```
   EVENTS build dependency graphs across streams:
     streamA:  kernel1 --record(e)-->
     streamB:            waitEvent(e) --> kernel2   (kernel2 starts after kernel1)
   This is the manual version of what CUDA Graphs (Ch. 16) automate.
```

---

## 6. Synchronization spectrum (know the cost)

```
   FINE-GRAINED (preferred)                                 COARSE (avoid in hot paths)
   cudaEventSynchronize(e)   cudaStreamSynchronize(s)   cudaDeviceSynchronize()
   wait one event            wait one stream            wait the WHOLE device
   (least blocking)                                     (blocks everything)

   Use stream/event sync so independent work keeps flowing. cudaDeviceSynchronize()
   is a big hammer — fine for teaching/debugging, bad for throughput.
```

```
   THE DEFAULT STREAM CAVEAT: the legacy default stream is "synchronizing" — it
   serializes with all other streams. To get concurrency, either use explicit
   non-default streams, or compile with --default-stream per-thread (each host
   thread gets its own concurrent default stream).
```

---

## 7. Stream-ordered memory allocation (modern, CUDA 11.2+)

`cudaMallocAsync`/`cudaFreeAsync` allocate and free *in stream order* from a
memory pool — fast, and they let allocation overlap and reuse across streams
without global synchronization (see Ch. 06).

```cpp
cudaMallocAsync(&ptr, bytes, s);     // allocation ordered within stream s
kernel<<<g, b, 0, s>>>(ptr, ...);
cudaFreeAsync(ptr, s);               // freed after prior work in s completes
```

---

## 8. Concurrency also comes from within one stream

Even a single stream benefits from the GPU running many blocks concurrently, and
multiple *small* kernels in different streams can run **side by side** on the SMs
if one kernel doesn't fill the GPU. This is how you keep a big GPU busy with
several modest kernels (common in inference serving).

```
   Kernel too small to fill the GPU?  Run several in different streams:
     stream A: small kernel  ┐
     stream B: small kernel  ├─ co-resident on the SMs -> better utilization
     stream C: small kernel  ┘
```

Run the worked example (with/without streams) to see the overlap on your GPU:

```bash
cd examples && make 07_streams_async && ./07_streams_async
```

---

## 9. Key takeaways

- Everything default is **serialized**; **streams** are ordered queues that run
  **concurrently** across each other when resources allow.
- **Pin host memory** (`cudaMallocHost`) — required for `cudaMemcpyAsync` to
  actually overlap.
- The **pipelining pattern** (chunk the data, one stream per chunk, async H2D →
  kernel → D2H) overlaps copies with compute, cutting time toward
  `max(copy, compute)` — often 2–3×.
- **Events** time GPU work and express cross-stream dependencies
  (`cudaStreamWaitEvent`).
- Prefer **fine-grained sync** (event/stream) over `cudaDeviceSynchronize`; watch
  the **default-stream** serialization caveat.
- Use **stream-ordered allocation** (`cudaMallocAsync`) for fast, overlap-friendly
  memory management; run several small kernels in streams to fill the GPU.

**Next:** [14 — Advanced kernel techniques →](14_advanced_kernel_techniques.md)
