# Overhead

A profiler that cannot state its own cost is asking to be trusted on faith. These numbers are measured, reproducible, and re-run every release.

```bash
./build/ring_stress --mode cost --records 20000000
```

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

**Not yet measured:** the CUPTI callback cost itself, which does not exist yet. When it lands it will almost certainly dominate everything in this table, and these totals will need restating rather than adding to.

## Correctness, and what is not yet proven

`./build/ring_stress --mode all` runs 44 checks across six modes: in-process streaming, a throttled reader against a 64-record ring (constant wrap and 98 % drops), the same across a real `fork` + `shm_open` pair, a producer SIGKILLed mid-loop, four attacks on the control block, and the cost breakdown. Every mode is clean under ThreadSanitizer, and 25 consecutive full runs showed no flakiness.

Two honest gaps:

**ThreadSanitizer cannot see the cross-process path.** It tracks shadow state per process, so in `--mode shm` it only ever observes the parent. The two-process protocol therefore has no automated ordering verification — the in-process modes exercise the same code and are TSan-clean, which is evidence but not proof.

**Everything here ran on x86-64,** which gives acquire-on-load and release-on-store almost for free. A missing `memory_order_release` would change no generated code on this machine and every test would still pass. The orderings have been reasoned through for a weak memory model and reviewed, but they have not been *run* on one. First ARM run is the real test.

## v0.1 — passive layer

NVML polling costs the observed processes nothing: it queries the driver from the agent's own process and never touches them. The agent's own cost is one NVML round trip per device per second, and it does not sample at all when no browser is connected.
