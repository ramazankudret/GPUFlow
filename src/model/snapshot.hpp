// The single contract between a collection layer and everything downstream.
//
// Both the NVML poller and the future CUPTI collector produce a Snapshot; the
// transport and the UI consume nothing else. Nothing in this header may name an
// NVML or CUPTI type — that isolation is what lets the two collection layers
// evolve independently.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpuflow {

// Sentinel for fields the driver declined to report. NVML returns
// NVML_VALUE_NOT_AVAILABLE for per-process memory under WDDM, and
// per-process utilization is unsupported on several virtualized setups,
// so "unknown" has to survive the trip to the UI rather than becoming a zero
// the operator would read as an idle process.
constexpr double kUnknownUtilization = -1.0;

struct ProcessSample {
    std::uint32_t pid = 0;
    std::string process_name;
    std::uint64_t used_gpu_memory_bytes = 0;
    bool memory_reported = false;
    double sm_utilization_percent = kUnknownUtilization;
};

struct GpuSample {
    std::uint32_t index = 0;
    std::string name;
    std::string uuid;
    std::uint32_t utilization_percent = 0;
    std::uint32_t memory_utilization_percent = 0;
    // Excludes the driver's own reservation, which is what nvidia-smi prints.
    // Disagreeing with the reference tool by 232 MiB on an idle card is how a
    // monitor loses an operator's trust.
    std::uint64_t memory_used_bytes = 0;
    std::uint64_t memory_reserved_bytes = 0;
    std::uint64_t memory_total_bytes = 0;

    // An empty process list means one of two very different things, and the
    // operator has to be able to tell them apart: nothing is using the card,
    // or this platform will not tell us. WSL2's passthrough shim is the second
    // case often enough that collapsing them would make the tool look broken.
    bool process_listing_supported = true;
    std::vector<ProcessSample> processes;
};

struct Snapshot {
    std::int64_t timestamp_unix_ms = 0;
    std::string driver_version;
    std::vector<GpuSample> gpus;

    // Hand-rolled; see snapshot.cpp for why no JSON library is pulled in.
    std::string to_json() const;
};

}  // namespace gpuflow
