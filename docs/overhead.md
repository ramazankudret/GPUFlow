# Overhead

A profiler that cannot state its own cost is asking to be trusted on faith. These numbers are measured, reproducible, and re-run every release.

```bash
./build/ring_stress --mode cost --records 20000000
./build/collector_probe -- ./your-cuda-program
```

## v0.2 — what the observed program actually pays

The figure that matters: a launch-bound CUDA program, run normally and then run again with the collector injected. Launch-bound on purpose — CUPTI instruments launches, so a compute-bound loop would hide the very cost being measured.

200 000 empty-kernel launches, five runs each way, same machine as below:

| | Per launch | Launches/s |
|---|---:|---:|
| Baseline | 7 579 ns | ~132 000 |
| With GPUFlow injected | 7 776 ns | ~129 000 |
| **Cost** | **~197 ns** | **2.6 %** |

Of that ~197 ns, roughly 30 ns is GPUFlow's own ring (measured separately below). The rest is CUPTI's activity instrumentation, which is the price of admission for kernel-level visibility without touching the program's source.

**Two things make that percentage flattering, and both should be said.** The baseline launch itself costs 7.6 µs here because this is WSL2, where every launch crosses a passthrough boundary; on native Linux a launch is nearer 2–5 µs, so the same ~197 ns would read as 4–8 %. And the kernels are empty — a program doing real work per launch amortises the overhead much further. Take 2.6 % as the number for *this* machine, not as a general claim.

Setting a smaller `GPUFLOW_FLUSH_MS` raises the cost and lowers latency; 100 ms is the default.

### What a crash costs

CUPTI buffers records and hands them back on a timer. If the observed program dies, everything since the last flush is gone. Measured: a program that launched 324 000 kernels over three seconds and then took SIGSEGV yielded 317 520 of them — **98 %**, with the loss confined to the final ~60 ms. That is the intended trade, and it is bounded by the flush period rather than by luck.

## v0.2 — event ring

## v0.2 — event ring

The transport the injected collector will write into. This is the only part that runs on the observed program's hot path; everything downstream happens in the agent's own process and costs the measured program nothing.

Measured on a 12th Gen Intel Core i7-12700H (10 cores / 20 threads), Linux 6.6 under WSL2, g++ 13.3 at `-O2`, with a consumer draining as fast as it can. Three runs, spread shown.

| Step | Cost | What it is |
|---|---:|---|
| Build event | 3.3 – 4.0 ns | Turning one activity record into a 64-byte `Event`, FNV checksum included. Pure register arithmetic. |
| `try_push` | 26 – 32 ns | The ring itself. |
| **Total** | **30 – 35 ns** | Per event, producer side. |

What that means for a program being watched:

| Kernel launch rate | Cost |
|---:|---:|
| 10 K/s | ~0.03 % of one core |
| 100 K/s | ~0.3 % of one core |
| 1 M/s | ~3 % of one core |

### Reading these numbers honestly

**`try_push` is dominated by cache coherence, not instructions.** The record is a 64-byte store into a line another core is pulling away. The instruction count is a handful of loads, a compare, a `memcpy`, and a release store; the ~26 ns is the line transfer. That makes the figure a ceiling — it was taken with the consumer spinning flat out, which maximises contention. An agent polling in batches contends far less, and the same benchmark with an idle consumer runs several times faster.

**The end-to-end streaming modes report a higher figure, 14–60 ns depending on the run, and that is not a contradiction.** `--mode threads` has the consumer verifying a checksum on every record as it drains — heavier than what the agent will do, and it keeps both cores fighting over the same lines continuously. The split above isolates the producer's own cost, which is the number that answers "what does GPUFlow cost the program it is measuring".

**The drop path is cheaper than the success path**, about 6.7 ns, because it never touches the record array. A saturated ring therefore degrades toward *less* overhead, not more — the collector cannot become the reason the observed program stalls.

**The ring is a small share of the total.** Of the ~197 ns the observed program pays per launch, this table accounts for about 30. That was the expected outcome and it is the right one: the transport should not be where the cost lives.

## Correctness, and what is not yet proven

`./build/ring_stress --mode all` runs 44 checks across six modes: in-process streaming, a throttled reader against a 64-record ring (constant wrap and 98 % drops), the same across a real `fork` + `shm_open` pair, a producer SIGKILLed mid-loop, four attacks on the control block, and the cost breakdown. Every mode is clean under ThreadSanitizer, and 25 consecutive full runs showed no flakiness.

Two honest gaps:

**ThreadSanitizer cannot see the cross-process path.** It tracks shadow state per process, so in `--mode shm` it only ever observes the parent. The two-process protocol therefore has no automated ordering verification — the in-process modes exercise the same code and are TSan-clean, which is evidence but not proof.

**Everything here ran on x86-64,** which gives acquire-on-load and release-on-store almost for free. A missing `memory_order_release` would change no generated code on this machine and every test would still pass. The orderings have been reasoned through for a weak memory model and reviewed, but they have not been *run* on one. First ARM run is the real test.

## v0.1 — passive layer

NVML polling costs the observed processes nothing: it queries the driver from the agent's own process and never touches them. The agent's own cost is one NVML round trip per device per second, and it does not sample at all when no browser is connected.
