// Lock-free single-producer / single-consumer ring over a shared mapping.
//
// This is the most safety-critical code in the project. The producer side runs
// inside someone else's process, on their hot path, and every rule below exists
// because breaking it makes GPUFlow the reason their program got slower or
// died:
//
//   - try_push() never blocks, never allocates, never takes a lock, and never
//     syscalls. Worst case it drops a record and increments a counter.
//   - A full ring drops the *newest* record. Overwriting the oldest would mean
//     writing into records the consumer may be mid-copy, and a profiler that
//     corrupts its own trace to keep up is worse than one that admits a gap.
//   - Drops are counted, not swallowed. The agent surfaces them so the UI can
//     show an honest hole rather than a continuous-looking timeline.
//
// The consumer side is the agent, reading a region another process can write at
// will — not necessarily a hostile one, but certainly a process whose stray
// pointer can land anywhere in this mapping. Two rules follow, and both are
// easy to violate by accident:
//
//   1. Geometry is validated ONCE, into RingGeometry, and every index is
//      derived from that copy. Nothing re-reads capacity from the shared page
//      after ring_validate returns — a value that can change between the check
//      and the use was never checked.
//   2. A state the protocol says is impossible means the ring is broken, not
//      that it holds a lot of records. pop_batch latches a fault and stops
//      rather than clamping and continuing; clamping would turn one bad store
//      into an endless stream of attacker-shaped records.
//
// SPSC is not a suggestion, in either direction. Two concurrent producers
// interleave into the same slot; two concurrent consumers hand out the same
// batch twice and then lose one. CUPTI delivers activity buffers on one worker
// thread, which is what makes the producer side safe — build with
// -DGPUFLOW_RING_DEBUG to abort the moment either rule is broken.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#include "ipc/event.hpp"

namespace gpuflow {

constexpr std::uint32_t kRingMagic = 0x47504652;  // 'GPFR'
constexpr std::uint32_t kRingVersion = 1;

// The record array starts on its own page so it never shares a cache line with
// the control block the two sides are hammering.
constexpr std::size_t kRingDataOffset = 4096;

constexpr std::size_t kCacheLine = 64;

// 64M records is 4 GiB of ring — far past anything useful, and low enough that
// capacity * sizeof(Event) cannot overflow a size_t on any target we accept.
constexpr std::uint32_t kMaxRingCapacity = 1u << 26;

static_assert(sizeof(std::size_t) >= 8, "ring sizing assumes a 64-bit size_t");

struct RingControl {
    // Immutable once published, but still living in a page another process can
    // write, so every one of these is atomic: a plain load here would be a
    // formal data race and the compiler would be free to reload it.
    //
    // magic is written last and read first. It is the flag that says the rest
    // of this block is filled in.
    std::atomic<std::uint32_t> magic;
    std::atomic<std::uint32_t> version;
    std::atomic<std::uint32_t> record_bytes;
    std::atomic<std::uint32_t> capacity;
    std::atomic<std::uint64_t> data_offset;

    // Producer-owned line. dropped shares it with head deliberately — same
    // writer, so there is no false sharing to avoid between them.
    alignas(kCacheLine) std::atomic<std::uint64_t> head;
    std::atomic<std::uint64_t> dropped;
    std::atomic<std::uint64_t> producer_pid;
    std::atomic<std::uint32_t> producer_attached;

    // Consumer-owned line.
    alignas(kCacheLine) std::atomic<std::uint64_t> tail;
    std::atomic<std::uint32_t> consumer_fault;
};

// Cross-process atomics only work if the atomic is the plain integer with no
// side table; a lock-based fallback would put the mutex in one process's heap.
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "64-bit atomics must be lock-free to cross a process boundary");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
              "32-bit atomics must be lock-free to cross a process boundary");

// Two independently-compiled binaries have to agree on this layout byte for
// byte, and `version` only helps if someone remembers to bump it. Pin the
// offsets so a reordering breaks the build instead of the protocol. Nothing
// conditional may ever be added to this struct — a debug-only member would
// give a debug agent and a release collector different offsets for `tail`.
static_assert(sizeof(RingControl) == 192, "RingControl layout is a cross-process ABI");
static_assert(offsetof(RingControl, head) == 64, "producer line moved");
static_assert(offsetof(RingControl, tail) == 128, "consumer line moved");
static_assert(sizeof(RingControl) <= kRingDataOffset, "control block must fit before the data page");

constexpr bool is_power_of_two(std::uint32_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

constexpr std::size_t ring_bytes(std::uint32_t capacity) noexcept {
    return kRingDataOffset + static_cast<std::size_t>(capacity) * sizeof(Event);
}

// The validated geometry, copied out of shared memory exactly once. Every index
// in this file is derived from one of these, never from RingControl.
struct RingGeometry {
    std::uint32_t capacity = 0;
    std::uint64_t mask = 0;
    bool ok = false;

    explicit operator bool() const noexcept { return ok; }
};

// Called by the agent, which owns the segment's lifecycle. The collector only
// ever opens an already-initialised ring — one less thing for injected code to
// get wrong inside a process it does not own.
inline void ring_init(void* base, std::uint32_t capacity) noexcept {
    std::memset(base, 0, kRingDataOffset);
    // Placement new so the creator side is a well-defined object rather than a
    // reinterpreted pile of bytes. The opener in the other process can only
    // reinterpret — that gap is inherent to shared memory and is why the layout
    // is all fixed-width types with pinned offsets.
    auto* c = ::new (base) RingControl();
    c->record_bytes.store(static_cast<std::uint32_t>(sizeof(Event)), std::memory_order_relaxed);
    c->capacity.store(capacity, std::memory_order_relaxed);
    c->data_offset.store(kRingDataOffset, std::memory_order_relaxed);
    c->head.store(0, std::memory_order_relaxed);
    c->tail.store(0, std::memory_order_relaxed);
    c->dropped.store(0, std::memory_order_relaxed);
    c->consumer_fault.store(0, std::memory_order_relaxed);
    c->version.store(kRingVersion, std::memory_order_relaxed);
    // Last, and with a release: anyone who acquires this magic sees everything
    // above it. A reader that arrives earlier sees zero and rejects the header.
    c->magic.store(kRingMagic, std::memory_order_release);
}

// The trust boundary. Loads each field exactly once and returns the geometry by
// value; callers must build their views from the returned struct, because the
// fields it came from can change the instant this returns.
inline RingGeometry ring_validate(const void* base, std::size_t mapped_bytes) noexcept {
    RingGeometry geo;
    if (base == nullptr || mapped_bytes < kRingDataOffset + sizeof(Event)) return geo;

    const auto* c = static_cast<const RingControl*>(base);
    if (c->magic.load(std::memory_order_acquire) != kRingMagic) return geo;
    if (c->version.load(std::memory_order_relaxed) != kRingVersion) return geo;
    if (c->record_bytes.load(std::memory_order_relaxed) != sizeof(Event)) return geo;
    if (c->data_offset.load(std::memory_order_relaxed) != kRingDataOffset) return geo;

    const std::uint32_t capacity = c->capacity.load(std::memory_order_relaxed);
    if (!is_power_of_two(capacity)) return geo;
    if (capacity > kMaxRingCapacity) return geo;
    if (ring_bytes(capacity) > mapped_bytes) return geo;

    geo.capacity = capacity;
    geo.mask = capacity - 1;
    geo.ok = true;
    return geo;
}

namespace detail {

inline Event* records_of(void* base) noexcept {
    // kRingDataOffset, not the shared data_offset field: validated-then-ignored
    // is the only safe way to treat geometry that another process can rewrite.
    return reinterpret_cast<Event*>(static_cast<char*>(base) + kRingDataOffset);
}

#ifdef GPUFLOW_RING_DEBUG
// Process-local, deliberately NOT in RingControl: an SPSC violation is two
// threads in one process, and putting the counter in shared memory would let a
// build flag change a cross-process ABI.
inline std::atomic<int>& producer_guard() noexcept {
    static std::atomic<int> g{0};
    return g;
}
inline std::atomic<int>& consumer_guard() noexcept {
    static std::atomic<int> g{0};
    return g;
}

class ExclusiveGuard {
public:
    explicit ExclusiveGuard(std::atomic<int>& counter) noexcept : counter_(counter) {
        if (counter_.fetch_add(1, std::memory_order_acq_rel) != 0) {
            // Silent interleaving is the one failure mode that would be
            // impossible to diagnose from a trace, so refuse to produce one.
            std::abort();
        }
    }
    ~ExclusiveGuard() { counter_.fetch_sub(1, std::memory_order_acq_rel); }
    ExclusiveGuard(const ExclusiveGuard&) = delete;
    ExclusiveGuard& operator=(const ExclusiveGuard&) = delete;

private:
    std::atomic<int>& counter_;
};
#endif

}  // namespace detail

// Producer view. Built from a validated geometry, never from the shared header.
class RingProducer {
public:
    RingProducer() noexcept = default;

    RingProducer(void* base, const RingGeometry& geo) noexcept
        : c_(geo.ok ? static_cast<RingControl*>(base) : nullptr),
          records_(geo.ok ? detail::records_of(base) : nullptr),
          mask_(geo.mask),
          capacity_(geo.capacity) {}

    bool valid() const noexcept { return c_ != nullptr; }

    // The hot path. No allocation, no lock, no syscall, no unbounded work.
    bool try_push(const Event& e) noexcept {
#ifdef GPUFLOW_RING_DEBUG
        detail::ExclusiveGuard guard(detail::producer_guard());
#endif
        const std::uint64_t head = c_->head.load(std::memory_order_relaxed);
        const std::uint64_t tail = c_->tail.load(std::memory_order_acquire);

        if (head - tail >= capacity_) {
            c_->dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        // memcpy rather than assignment: these bytes are a shared mapping, not
        // an Event whose lifetime this process began.
        std::memcpy(&records_[head & mask_], &e, sizeof(Event));

        // Release pairs with the consumer's acquire on head — the record is
        // fully visible before the index that publishes it.
        c_->head.store(head + 1, std::memory_order_release);
        return true;
    }

    std::uint64_t dropped() const noexcept {
        return c_->dropped.load(std::memory_order_relaxed);
    }
    std::uint64_t produced() const noexcept {
        return c_->head.load(std::memory_order_relaxed);
    }

private:
    RingControl* c_ = nullptr;
    Event* records_ = nullptr;
    std::uint64_t mask_ = 0;
    std::uint64_t capacity_ = 0;
};

// A coherent read of the counters. Taken in one place because the orderings
// only compose in this sequence: acquiring head first is what makes the
// producer's earlier relaxed writes to dropped visible. Reading the accessors
// in some other order can, on a weak memory model, report a drop count that
// predates records already consumed — and "an honest gap" is the whole point.
struct RingStats {
    std::uint64_t produced = 0;
    std::uint64_t consumed = 0;
    std::uint64_t dropped = 0;
    bool faulted = false;
};

// Consumer view — the agent side.
class RingConsumer {
public:
    RingConsumer() noexcept = default;

    RingConsumer(void* base, const RingGeometry& geo) noexcept
        : c_(geo.ok ? static_cast<RingControl*>(base) : nullptr),
          records_(geo.ok ? detail::records_of(base) : nullptr),
          mask_(geo.mask),
          capacity_(geo.capacity) {}

    bool valid() const noexcept { return c_ != nullptr; }

    // Returns 0 forever once the ring has been declared broken.
    std::size_t pop_batch(Event* out, std::size_t max) noexcept {
#ifdef GPUFLOW_RING_DEBUG
        detail::ExclusiveGuard guard(detail::consumer_guard());
#endif
        if (c_ == nullptr || faulted()) return 0;

        const std::uint64_t tail = c_->tail.load(std::memory_order_relaxed);
        const std::uint64_t head = c_->head.load(std::memory_order_acquire);

        const std::uint64_t available = head - tail;
        if (available > capacity_) {
            // Unreachable by the protocol: try_push only advances head while
            // head - tail < capacity, and tail is monotone. Getting here means
            // the control block was corrupted — possibly by a head stored
            // *behind* tail, which underflows to a huge value. Clamping and
            // carrying on would hand the agent an endless stream of whatever
            // happens to be in the mapping, and would march tail past head so
            // the ring could never recover. Stop instead, and say so.
            fault();
            return 0;
        }
        if (available == 0) return 0;

        const std::size_t n = available < max ? static_cast<std::size_t>(available) : max;
        for (std::size_t i = 0; i < n; ++i) {
            std::memcpy(&out[i], &records_[(tail + i) & mask_], sizeof(Event));
        }

        // Release so the producer cannot reuse these slots until the copies
        // above have actually happened.
        c_->tail.store(tail + n, std::memory_order_release);
        return n;
    }

    RingStats stats() const noexcept {
        RingStats s;
        if (c_ == nullptr) return s;
        s.produced = c_->head.load(std::memory_order_acquire);
        s.dropped = c_->dropped.load(std::memory_order_relaxed);
        s.consumed = c_->tail.load(std::memory_order_relaxed);
        s.faulted = c_->consumer_fault.load(std::memory_order_relaxed) != 0;
        return s;
    }

    bool faulted() const noexcept {
        return c_->consumer_fault.load(std::memory_order_relaxed) != 0;
    }

    std::uint32_t capacity() const noexcept { return static_cast<std::uint32_t>(capacity_); }

    // Advisory only. A collector killed by a signal never gets to clear this,
    // so the agent has to learn the child is gone from waitpid, not from here.
    bool producer_attached() const noexcept {
        return c_ != nullptr && c_->producer_attached.load(std::memory_order_acquire) != 0;
    }

private:
    void fault() noexcept { c_->consumer_fault.store(1, std::memory_order_relaxed); }

    RingControl* c_ = nullptr;
    Event* records_ = nullptr;
    std::uint64_t mask_ = 0;
    std::uint64_t capacity_ = 0;
};

}  // namespace gpuflow
