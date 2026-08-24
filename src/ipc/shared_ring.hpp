// shm_open/mmap lifecycle for the event ring.
//
// The agent creates the segment and unlinks it; the injected collector only
// ever opens one. That split is deliberate. The collector lives inside a
// process it does not own and may be killed at any moment — giving it the
// lifecycle would leave orphaned segments in /dev/shm every time the observed
// program crashed. The agent outlives the child it launched, so ownership sits
// with the party that can actually clean up.
//
// The segment is created 0600. It carries one user's GPU activity in detail,
// and nothing about it should be readable by other accounts on a shared box.

#pragma once

#include <cstdint>
#include <string>

#include "ipc/spsc_ring.hpp"

namespace gpuflow {

// Set by `gpuflow run` alongside CUDA_INJECTION64_PATH so the collector knows
// which segment to attach to.
constexpr const char* kRingEnvVar = "GPUFLOW_RING";

std::string ring_name_for_pid(std::uint64_t pid);

class SharedRing {
public:
    SharedRing() noexcept = default;
    ~SharedRing();

    SharedRing(SharedRing&& other) noexcept;
    SharedRing& operator=(SharedRing&& other) noexcept;
    SharedRing(const SharedRing&) = delete;
    SharedRing& operator=(const SharedRing&) = delete;

    // Agent side. Creates, sizes, maps, and initialises the ring; unlinks it on
    // destruction. Throws std::runtime_error on failure.
    static SharedRing create(const std::string& name, std::uint32_t capacity);

    // Collector side. Opens an existing ring and validates its geometry before
    // returning. Throws std::runtime_error if the segment is missing or its
    // header does not describe a ring this build can read.
    static SharedRing open(const std::string& name);

    bool valid() const noexcept { return base_ != nullptr && geometry_.ok; }
    void* base() const noexcept { return base_; }
    std::size_t mapped_bytes() const noexcept { return bytes_; }
    const std::string& name() const noexcept { return name_; }
    const RingGeometry& geometry() const noexcept { return geometry_; }

    // Both views are built from the geometry captured at validation. Nothing
    // here re-reads capacity from the mapping — see the note in spsc_ring.hpp
    // on why a value that can change after the check was never checked.
    RingProducer producer() const noexcept { return RingProducer(base_, geometry_); }
    RingConsumer consumer() const noexcept { return RingConsumer(base_, geometry_); }

    // Collector announces itself so the agent can tell "not started yet" from
    // "started and quiet" — the same distinction the passive layer makes
    // between an unsupported platform and an idle one.
    void mark_producer_attached(std::uint64_t pid) noexcept;
    void mark_producer_detached() noexcept;

private:
    void reset() noexcept;

    void* base_ = nullptr;
    std::size_t bytes_ = 0;
    RingGeometry geometry_;
    std::string name_;
    bool owns_ = false;  // only the creator unlinks
};

}  // namespace gpuflow
