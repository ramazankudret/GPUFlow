// libgpuflow_inject.so — the collector that runs inside someone else's process.
//
// The CUDA driver loads this library at CUDA initialisation because the agent
// put its path in CUDA_INJECTION64_PATH, and calls InitializeInjection(). From
// that moment everything here executes inside a program GPUFlow does not own,
// and the constraints in CLAUDE.md are not style preferences:
//
//   - No sockets. Ever. Events go into a shared-memory ring and nothing else
//     leaves this process. The agent reads that ring from outside.
//   - Nothing this file does may abort, throw past the callback boundary, or
//     block. If the ring is missing or malformed, the collector disables
//     itself and the observed program runs exactly as if GPUFlow were absent.
//   - No allocation on the record path. Buffers are taken from a pool sized at
//     startup; the only allocation after init is interning a kernel name the
//     first time it is seen, which is bounded by the program's kernel count.
//
// A note on where the cost lands: this code never runs on the application's
// launch path. CUPTI records activity into its own buffers and hands them back
// on a worker thread, which is where bufferCompleted below executes. The
// launch-path cost belongs to CUPTI's instrumentation, not to GPUFlow, and the
// two are measured separately in docs/overhead.md.
//
// Activity struct versions are pinned to older revisions on purpose. CUPTI
// appends fields rather than reordering them, so an older struct is a prefix of
// a newer record and stays readable when the runtime is ahead of these headers
// — the direction that actually happens in the field.

#include <cupti.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "ipc/name_table.hpp"
#include "ipc/shared_ring.hpp"
#include "ipc/spsc_ring.hpp"

namespace gpuflow {
namespace {

// CUPTI hands back buffers in batches; a few large ones beat many small ones.
constexpr std::size_t kBufferBytes = 256 * 1024;
constexpr std::size_t kBufferPool = 8;
constexpr std::size_t kBufferAlign = 8;

struct Collector {
    SharedRing ring;
    SharedRing names;
    RingProducer producer;
    NameWriter interner;
    bool active = false;

    // Pre-allocated so bufferRequested never calls into the allocator while
    // the observed program is running.
    std::vector<std::uint8_t*> pool;
    std::atomic<std::size_t> pool_top{0};

    std::uint64_t sequence = 0;
    std::uint64_t dropped_records = 0;
};

Collector& collector() {
    static Collector c;
    return c;
}

// Diagnostics go to stderr and only when asked. A profiler that prints into
// someone else's stdout has already broken their program.
bool verbose() {
    static const bool v = ::getenv("GPUFLOW_VERBOSE") != nullptr;
    return v;
}

void note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void note(const char* fmt, ...) {
    if (!verbose()) return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[gpuflow] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

bool cupti_ok(CUptiResult r, const char* what) {
    if (r == CUPTI_SUCCESS) return true;
    const char* msg = nullptr;
    cuptiGetResultString(r, &msg);
    note("%s failed: %s (%d)", what, msg ? msg : "?", static_cast<int>(r));
    return false;
}

/* --------------------------------------------------------------- buffers */

void CUPTIAPI buffer_requested(std::uint8_t** buffer, std::size_t* size,
                               std::size_t* max_num_records) {
    Collector& c = collector();
    *size = kBufferBytes;
    *max_num_records = 0;  // let CUPTI fill the buffer

    const std::size_t top = c.pool_top.load(std::memory_order_relaxed);
    if (top > 0) {
        c.pool_top.store(top - 1, std::memory_order_relaxed);
        *buffer = c.pool[top - 1];
        return;
    }

    // Pool exhausted. Falling back to the allocator here is deliberate: the
    // alternative is handing CUPTI a null buffer, which makes it drop activity
    // records wholesale. This path is rare and off the application's hot path.
    *buffer = static_cast<std::uint8_t*>(std::aligned_alloc(kBufferAlign, kBufferBytes));
    if (*buffer == nullptr) *size = 0;
}

void release_buffer(std::uint8_t* buffer) {
    Collector& c = collector();
    const std::size_t top = c.pool_top.load(std::memory_order_relaxed);
    if (top < c.pool.size()) {
        c.pool[top] = buffer;
        c.pool_top.store(top + 1, std::memory_order_relaxed);
        return;
    }
    std::free(buffer);
}

/* --------------------------------------------------------- record mapping */

std::uint8_t memcpy_flags(std::uint8_t copy_kind) {
    switch (copy_kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return kFlagHostToDevice;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return kFlagDeviceToHost;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return kFlagDeviceToDevice;
        default: return 0;
    }
}

void push(Collector& c, Event& e) {
    e.sequence = c.sequence++;
    event_seal(e);
    if (!c.producer.try_push(e)) ++c.dropped_records;
}

void handle_kernel(Collector& c, const CUpti_ActivityKernel4* k) {
    Event e{};
    e.kind = static_cast<std::uint8_t>(EventKind::kKernel);
    e.device_id = static_cast<std::uint16_t>(k->deviceId);
    e.stream_id = k->streamId;
    e.start_ns = k->start;
    e.end_ns = k->end;
    e.correlation_id = k->correlationId;
    e.name_id = c.interner.intern(k->name);
    e.value_a = static_cast<std::uint64_t>(k->gridX) * k->gridY * k->gridZ;
    e.value_b = static_cast<std::uint64_t>(k->blockX) * k->blockY * k->blockZ;
    e.context_id = k->contextId;
    push(c, e);
}

void handle_memcpy(Collector& c, const CUpti_ActivityMemcpy3* m) {
    Event e{};
    e.kind = static_cast<std::uint8_t>(EventKind::kMemcpy);
    e.flags = memcpy_flags(m->copyKind);
    e.device_id = static_cast<std::uint16_t>(m->deviceId);
    e.stream_id = m->streamId;
    e.start_ns = m->start;
    e.end_ns = m->end;
    e.correlation_id = m->correlationId;
    e.name_id = 0;
    e.value_a = m->bytes;
    e.context_id = m->contextId;
    push(c, e);
}

void CUPTIAPI buffer_completed(CUcontext, std::uint32_t, std::uint8_t* buffer, std::size_t,
                               std::size_t valid_bytes) {
    Collector& c = collector();
    if (!c.active) {
        release_buffer(buffer);
        return;
    }

    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_bytes, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL:
                handle_kernel(c, reinterpret_cast<const CUpti_ActivityKernel4*>(record));
                break;
            case CUPTI_ACTIVITY_KIND_MEMCPY:
                handle_memcpy(c, reinterpret_cast<const CUpti_ActivityMemcpy3*>(record));
                break;
            default:
                break;
        }
    }

    std::size_t cupti_dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &cupti_dropped) == CUPTI_SUCCESS &&
        cupti_dropped > 0) {
        // CUPTI's own losses, distinct from ours. Both end up as gaps in the
        // timeline and the operator deserves to know which is which.
        note("CUPTI dropped %zu records", cupti_dropped);
    }

    release_buffer(buffer);
}

/* ------------------------------------------------------------- lifecycle */

void shutdown() {
    Collector& c = collector();
    if (!c.active) return;

    // Flush before going inactive, not after. The flush is what delivers every
    // record CUPTI is still holding — clearing the flag first makes
    // buffer_completed discard exactly the batch this call exists to collect,
    // and the run ends with an empty ring and no error anywhere.
    cuptiActivityFlushAll(1);
    c.active = false;

    c.ring.mark_producer_detached();
    note("detached: %llu events, %llu dropped by the ring",
         static_cast<unsigned long long>(c.sequence),
         static_cast<unsigned long long>(c.dropped_records));
}

bool start() {
    Collector& c = collector();

    const char* ring_name = ::getenv(kRingEnvVar);
    if (ring_name == nullptr) {
        note("%s not set — collector idle", kRingEnvVar);
        return false;
    }

    // Every failure below leaves the observed program running normally. There
    // is no configuration of this library that is allowed to be fatal to it.
    try {
        c.ring = SharedRing::open(ring_name);
    } catch (const std::exception& e) {
        note("cannot attach to ring: %s", e.what());
        return false;
    }
    c.producer = c.ring.producer();

    if (const char* names_name = ::getenv(kNameTableEnvVar)) {
        try {
            c.names = SharedRing::open_raw(names_name);
            const NameTableGeometry geo =
                name_table_validate(c.names.base(), c.names.mapped_bytes());
            if (geo) c.interner = NameWriter(c.names.base(), geo);
        } catch (const std::exception& e) {
            note("cannot attach to name table: %s — kernels will be unnamed", e.what());
        }
    }

    c.pool.resize(kBufferPool, nullptr);
    for (std::size_t i = 0; i < kBufferPool; ++i) {
        auto* b = static_cast<std::uint8_t*>(std::aligned_alloc(kBufferAlign, kBufferBytes));
        if (b == nullptr) break;
        c.pool[i] = b;
        c.pool_top.store(i + 1, std::memory_order_relaxed);
    }

    if (!cupti_ok(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed),
                  "cuptiActivityRegisterCallbacks")) {
        return false;
    }

    // Kernels and copies only. Every extra activity kind is more work on
    // someone else's machine for data v0.2 does not draw.
    cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
             "enable CONCURRENT_KERNEL");
    cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "enable MEMCPY");

    // Without this, CUPTI only hands buffers back when they fill or at the
    // final flush — which would make a *live* profiler show nothing until the
    // program exits, and show nothing at all if it crashes instead. CUPTI runs
    // the timer on its own worker, so this costs the collector no thread of its
    // own inside a process it does not own.
    //
    // The period is the trade: shorter means fresher lanes in the UI and less
    // lost when the observed program dies, longer means fewer interruptions for
    // the program being measured.
    std::uint32_t flush_ms = 100;
    if (const char* v = ::getenv("GPUFLOW_FLUSH_MS")) {
        const long parsed = std::strtol(v, nullptr, 10);
        if (parsed >= 10 && parsed <= 10000) flush_ms = static_cast<std::uint32_t>(parsed);
    }
    cupti_ok(cuptiActivityFlushPeriod(flush_ms), "cuptiActivityFlushPeriod");

    c.ring.mark_producer_attached(static_cast<std::uint64_t>(::getpid()));
    c.active = true;

    std::uint32_t version = 0;
    cuptiGetVersion(&version);
    note("attached to %s (pid %d, CUPTI runtime %u, headers %d, flush every %u ms)", ring_name,
         ::getpid(), version, CUPTI_API_VERSION, flush_ms);

    std::atexit(shutdown);
    return true;
}

}  // namespace
}  // namespace gpuflow

// The driver's entry point. Returning zero aborts CUDA initialisation in the
// host program, so this returns 1 even when the collector could not start —
// a GPUFlow that cannot attach must be invisible, not fatal.
extern "C" int InitializeInjection(void) {
    gpuflow::start();
    return 1;
}
