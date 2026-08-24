// The wire record the injected collector writes and the agent reads.
//
// Fixed 64 bytes — one cache line, power of two, so the ring indexes by mask
// and never has to reason about a record straddling the wrap. Variable-length
// data (kernel names) cannot live here: the collector runs on someone else's
// hot path and is not allowed to allocate. Names are interned once and
// referenced by `name_id`; a training loop launches the same twenty kernels a
// million times, so the table stays small and the hot path stays a memcpy.
//
// Every field is written by another process. The agent must treat all of it as
// untrusted input — `check` catches a torn or corrupted record, but a hostile
// producer can compute a valid one, so `name_id` and the rest still need bounds
// checks at the point of use.

#pragma once

#include <cstddef>
#include <cstdint>

namespace gpuflow {

enum class EventKind : std::uint8_t {
    kNone = 0,
    kKernel = 1,
    kMemcpy = 2,
    kStreamCreated = 3,
    kStreamDestroyed = 4,
    kSynthetic = 0xff,  // stress harness only; the agent ignores these
};

// Memcpy direction, carried in `flags` when kind == kMemcpy.
enum EventFlags : std::uint8_t {
    kFlagHostToDevice = 1u << 0,
    kFlagDeviceToHost = 1u << 1,
    kFlagDeviceToDevice = 1u << 2,
    kFlagAsync = 1u << 3,
};

struct Event {
    std::uint8_t kind;
    std::uint8_t flags;
    std::uint16_t device_id;
    std::uint32_t stream_id;

    // CUPTI's clock, not the agent's. Correlating the two is a separate job;
    // storing the raw value keeps that correction out of the hot path.
    std::uint64_t start_ns;
    std::uint64_t end_ns;

    std::uint32_t correlation_id;
    std::uint32_t name_id;

    std::uint64_t value_a;  // kernel: grid size      memcpy: bytes
    std::uint64_t value_b;  // kernel: block size     memcpy: reserved

    // Producer-monotonic. The ring's own indices already order records, but an
    // explicit sequence lets the reader prove nothing was lost or reordered
    // without trusting the indices it is validating.
    std::uint64_t sequence;

    std::uint32_t context_id;
    std::uint32_t check;
};

static_assert(sizeof(Event) == 64, "Event must stay one cache line");
static_assert(alignof(Event) <= 64, "Event must not over-align the ring array");

// Cheap enough to sit on the hot path: the fields are already in registers, and
// it turns a torn record into a detectable one instead of a plausible one.
inline std::uint32_t event_check(const Event& e) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto fold = [&h](std::uint64_t v) noexcept {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    fold(static_cast<std::uint64_t>(e.kind) | (static_cast<std::uint64_t>(e.flags) << 8) |
         (static_cast<std::uint64_t>(e.device_id) << 16) |
         (static_cast<std::uint64_t>(e.stream_id) << 32));
    fold(e.start_ns);
    fold(e.end_ns);
    fold(static_cast<std::uint64_t>(e.correlation_id) |
         (static_cast<std::uint64_t>(e.name_id) << 32));
    fold(e.value_a);
    fold(e.value_b);
    fold(e.sequence);
    fold(e.context_id);
    return static_cast<std::uint32_t>(h ^ (h >> 32));
}

inline void event_seal(Event& e) noexcept {
    e.check = event_check(e);
}

inline bool event_verify(const Event& e) noexcept {
    return e.check == event_check(e);
}

}  // namespace gpuflow
