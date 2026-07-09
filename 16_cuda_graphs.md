# 16 — CUDA Graphs

> Part of **[CUDA Know-Hows](README.md)**. Prev: [15 — Advanced memory techniques](15_advanced_memory_techniques.md).
> Next: [17 — Multi-GPU](17_multi_gpu.md).
>
> Goal: eliminate per-launch CPU overhead for repeated work by recording a
> sequence of operations *once* as a graph and replaying it many times. Essential
> for small-kernel-heavy workloads (deep-learning inference, iterative solvers)
> where launch latency dominates.

---

## 1. The problem: launch overhead dominates small kernels

Each kernel launch costs a few microseconds of CPU-side work (argument marshaling,
driver calls, stream bookkeeping). If a kernel runs for 500 µs, a 5 µs launch is
noise. But real pipelines fire **hundreds of tiny kernels per iteration**, and
then launch overhead can exceed the actual compute.

```
   STREAM launches (per iteration): CPU issues each op, one at a time
     CPU:  [launch K1][launch K2][launch K3]...[launch K200]   <- 200x overhead
     GPU:      [K1][gap][K2][gap][K3]...                       <- gaps = CPU can't
                                                                  feed GPU fast enough

   CUDA GRAPH: record the 200 ops ONCE; replay with a SINGLE launch call
     CPU:  [graphLaunch]                                       <- one call
     GPU:  [K1][K2][K3]...[K200]                               <- back-to-back, no gaps
```

```
   A CUDA graph is a DAG of operations (kernels, memcpys, memsets, host callbacks,
   even child graphs) with their dependencies. You define it once (topology +
   params), the driver optimizes/validates it once, then each replay skips almost
   all per-op CPU overhead. Typical win: 2x+ on launch-bound, many-small-kernel
   workloads.
```

---

## 2. Building a graph — the easy way: stream capture

The simplest path: wrap your existing stream code in begin/end capture. CUDA
records the operations and their dependencies into a graph automatically.

```cpp
cudaGraph_t graph;
cudaGraphExec_t exec;

// 1. CAPTURE: record the stream's operations into a graph (nothing runs yet)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernelA<<<g, b, 0, stream>>>(...);
    kernelB<<<g, b, 0, stream>>>(...);
    cudaMemcpyAsync(..., stream);
    kernelC<<<g, b, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);

// 2. INSTANTIATE: compile the graph into an executable form (validate/optimize once)
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// 3. LAUNCH: replay the whole thing with one call, as many times as you like
for (int i = 0; i < numIterations; ++i)
    cudaGraphLaunch(exec, stream);
cudaStreamSynchronize(stream);

cudaGraphExecDestroy(exec);
cudaGraphDestroy(graph);
```

```
   THREE PHASES — do the expensive ones ONCE:
     CAPTURE     : record topology + params        (per-graph, once)
     INSTANTIATE : validate + optimize -> exec      (per-graph, once — not cheap!)
     LAUNCH      : replay                            (per-iteration, very cheap)
   The whole point is amortizing capture+instantiate over MANY launches. Don't
   rebuild the graph every iteration.
```

---

## 3. Building a graph — the explicit API

For full control (or when capture is awkward), build nodes and dependencies by
hand:

```cpp
cudaGraph_t graph; cudaGraphCreate(&graph, 0);
cudaGraphNode_t a, b;
cudaKernelNodeParams pa = {...};
cudaGraphAddKernelNode(&a, graph, nullptr, 0, &pa);      // no dependencies
cudaKernelNodeParams pb = {...};
cudaGraphAddKernelNode(&b, graph, &a, 1, &pb);           // b depends on a
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
```

```
   EXPLICIT GRAPH (a DAG):
        ┌──▶ [kernel B] ──┐
   [A] ─┤                 ├─▶ [D]     A runs; B and C run concurrently after A;
        └──▶ [kernel C] ──┘           D runs after both B and C.
   The graph EXPRESSES the parallelism; the runtime schedules B and C together.
```

---

## 4. Updating a graph without re-instantiating

Re-instantiation is expensive, so when only *parameters* change (pointers, sizes,
kernel args) between replays, update in place:

```cpp
cudaGraphExecKernelNodeSetParams(exec, node, &newParams);   // change one node's args
// or update everything changed since capture:
cudaGraphExecUpdate(exec, newGraph, &updateResult, ...);    // topology unchanged
```

```
   UPDATE vs RE-INSTANTIATE:
     same topology, new params -> cudaGraphExecUpdate / *SetParams  (cheap)
     topology changed          -> must re-instantiate               (expensive)
   Design your graph so the shape is stable and only data pointers/sizes vary.
```

---

## 5. Conditional & dynamic graphs (CUDA 12.3+ / 12.8+)

Modern CUDA adds **conditional nodes** — IF, WHILE, and (12.8+) SWITCH — so a
graph can contain data-dependent control flow without returning to the host.
This lets iterative algorithms (convergence loops, dynamic batching) live
entirely inside a replayable graph.

```
   CONDITIONAL NODES let the GPU decide control flow inside the graph:
     [setup] ─▶ [IF converged?] ──no──▶ [iterate] ──▶ (loop back / WHILE)
                      │yes
                      ▼
                  [finalize]
   Previously this required host round-trips between launches; now it stays on-GPU.
```

---

## 6. When to use graphs (and when not to)

```
   USE CUDA GRAPHS WHEN:
     - you launch the SAME sequence of ops repeatedly (training step, inference,
       time-stepping solver)
     - many SMALL kernels where launch overhead is a real fraction of time
     - you want deterministic, low-jitter replay

   DON'T BOTHER WHEN:
     - a few large kernels (launch overhead is negligible)
     - the topology changes every iteration (re-instantiate cost eats the benefit)
     - one-shot work that runs once

   INTEGRATION: frameworks use graphs under the hood — PyTorch `cuda.CUDAGraph` /
   `torch.compile`'s CUDA graph mode, TensorRT, etc. Capturing a model's steady-
   state step into a graph is a standard inference-latency optimization.
```

---

## 7. Key takeaways

- CUDA graphs record a **DAG of operations once** and **replay** it with minimal
  CPU overhead — the fix for launch-bound, many-small-kernel workloads.
- Build them via **stream capture** (easiest) or the **explicit node API** (full
  control); phases are **capture → instantiate (once, costly) → launch (cheap,
  repeated)**.
- Keep the **topology stable** and change only parameters with
  `cudaGraphExecUpdate` / `*SetParams` to avoid re-instantiation.
- **Conditional nodes** (IF/WHILE/SWITCH) keep data-dependent control flow on the
  GPU.
- Use graphs for **repeated** sequences (training/inference/solvers), not for a
  few big or ever-changing kernels. Frameworks already use them — capturing your
  steady-state step is a standard latency win.

**Next:** [17 — Multi-GPU →](17_multi_gpu.md)
