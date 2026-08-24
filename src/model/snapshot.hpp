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

// Per (kernel name, stream). Always sent in full: the number of distinct
// kernels a program has is small and does not grow with how hard it runs them,
// which is exactly the property the raw spans lack.
struct KernelStat {
    std::uint32_t name_index = 0;
    std::uint32_t stream_id = 0;
    std::uint64_t launches = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t max_ns = 0;
};

// One kernel execution, for the lanes. These are what cannot all be sent: a
// program launching 100k kernels a second would be 100k of these per frame,
// which is neither transmittable nor drawable. A bounded, most-recent sample
// goes out and TracedProcess::spans_elided says how much did not.
struct KernelSpan {
    std::uint32_t stream_id = 0;
    std::uint32_t name_index = 0;
    std::int64_t start_offset_us = 0;  // relative to the snapshot timestamp
    std::uint32_t duration_ns = 0;

    // 0 = kernel, 1 = host-to-device, 2 = device-to-host, 3 = device-to-device.
    // A copy stream carries no kernels, so without this the lane a program
    // dedicates to transfers would not exist at all in the plane — which is
    // exactly the structure v0.2 exists to show.
    std::uint8_t kind = 0;
};

// CUDA gives streams numbers, not names. `name_index` is non-zero only when the
// observed program named the stream through NVTX; `is_default` is CUPTI's own
// report of which id the context uses for the null stream. Everything else is
// an id, and is drawn as one — inferring a stream's purpose from what happened
// to run on it would be a guess wearing a label.
struct StreamInfo {
    std::uint32_t id = 0;
    std::uint32_t name_index = 0;
    bool is_default = false;
};

struct CopyStat {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
    std::uint64_t total_ns = 0;
};

// A process GPUFlow launched and is instrumenting. Absent for every process the
// passive layer merely observes — the distinction is the whole shape of the
// tool, and the UI has to be able to see it.
struct TracedProcess {
    std::uint32_t pid = 0;
    std::string command;
    bool running = false;

    std::vector<std::string> names;  // indexed by KernelStat/KernelSpan
    std::vector<StreamInfo> streams;
    std::vector<KernelStat> kernels;
    std::vector<KernelSpan> spans;

    CopyStat host_to_device;
    CopyStat device_to_host;
    CopyStat device_to_device;

    std::uint64_t events_total = 0;
    std::uint64_t events_dropped = 0;   // the ring was full
    std::uint64_t spans_elided = 0;     // in the window but not sent
    std::int64_t clock_offset_ns = 0;   // CUPTI's clock minus the agent's
};

struct Snapshot {
    std::int64_t timestamp_unix_ms = 0;
    std::string driver_version;
    std::vector<GpuSample> gpus;
    std::vector<TracedProcess> traced;

    // Hand-rolled; see snapshot.cpp for why no JSON library is pulled in.
    std::string to_json() const;
};

}  // namespace gpuflow
