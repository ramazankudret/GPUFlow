// NVML polling. The only file in the project that includes nvml.h.
//
// Two things here are load-bearing and easy to get wrong:
//
// 1. The running-process calls exist as _v1, _v2 and _v3 across two different
//    struct layouts, and the header we compile against can legitimately
//    disagree with the driver we run against — WSL2's passthrough shim is
//    exactly that case. They are resolved through dlsym at construction,
//    newest first, so the choice follows the runtime library rather than
//    whatever macro the header happened to pick. Every other NVML call used
//    here has been ABI-stable for a decade and is linked normally.
//
// 2. A failure inside poll() must never take the stream down. NVML declines
//    calls per-device and per-feature (WDDM hides per-process memory; several
//    virtualized setups have no per-process utilization at all), and the right
//    answer is a degraded field, not a dead agent.

#include "nvml/device_monitor.hpp"

#include <nvml.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpuflow {
namespace {

// _v3's nvmlProcessInfo_t and nvmlProcessInfo_v2_t are the same four fields in
// the same order, so one signature covers both. _v1 predates the MIG fields and
// needs its own.
using ProcessQueryWide = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_v2_t*);
using ProcessQueryNarrow = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_v1_t*);

// Casting void* to a function pointer is only conditionally supported by the
// standard; memcpy through the object representation is the portable spelling
// and keeps -Wpedantic quiet.
template <typename Fn>
Fn function_cast(void* symbol) {
    Fn fn = nullptr;
    std::memcpy(&fn, &symbol, sizeof(fn));
    return fn;
}

struct ProcessQuery {
    ProcessQueryWide wide = nullptr;
    ProcessQueryNarrow narrow = nullptr;
    const char* resolved_symbol = nullptr;

    bool available() const { return wide != nullptr || narrow != nullptr; }
};

// The other call whose ABI actually moved. v1 folds the driver's reservation
// into `used`; v2 breaks it out, and v2's `used` is the figure nvidia-smi
// prints — so this is the difference between agreeing with the reference tool
// and looking wrong by a couple of hundred megabytes.
using MemoryQueryV2 = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_v2_t*);

// RTLD_DEFAULT searches the already-loaded libnvidia-ml, so this reports what
// the driver actually exports rather than what the build-time header declared.
ProcessQuery resolve_process_query(const std::string& base) {
    ProcessQuery query;

    const std::string v3 = base + "_v3";
    if (void* symbol = dlsym(RTLD_DEFAULT, v3.c_str())) {
        query.wide = function_cast<ProcessQueryWide>(symbol);
        query.resolved_symbol = "_v3";
        return query;
    }

    const std::string v2 = base + "_v2";
    if (void* symbol = dlsym(RTLD_DEFAULT, v2.c_str())) {
        query.wide = function_cast<ProcessQueryWide>(symbol);
        query.resolved_symbol = "_v2";
        return query;
    }

    if (void* symbol = dlsym(RTLD_DEFAULT, base.c_str())) {
        query.narrow = function_cast<ProcessQueryNarrow>(symbol);
        query.resolved_symbol = "_v1";
    }
    return query;
}

struct RawProcess {
    std::uint32_t pid = 0;
    std::uint64_t used_gpu_memory_bytes = 0;
    bool memory_reported = false;
};

// NVML's count-out convention needs a retry when the process list grew between
// the sizing call and the fetch. Bounded, because an unbounded retry against a
// driver that keeps returning INSUFFICIENT_SIZE would hang the poll loop.
constexpr int kMaxSizingAttempts = 4;
constexpr unsigned int kInitialProcessCapacity = 16;

template <typename Info, typename Fn>
nvmlReturn_t fetch_processes(Fn fn, nvmlDevice_t device, std::vector<Info>& infos,
                             unsigned int& count) {
    infos.assign(kInitialProcessCapacity, Info{});
    for (int attempt = 0; attempt < kMaxSizingAttempts; ++attempt) {
        count = static_cast<unsigned int>(infos.size());
        const nvmlReturn_t rc = fn(device, &count, infos.data());
        if (rc != NVML_ERROR_INSUFFICIENT_SIZE) {
            return rc;
        }
        infos.assign(count + 8, Info{});
    }
    return NVML_ERROR_INSUFFICIENT_SIZE;
}

// NVML reports usedGpuMemory as NVML_VALUE_NOT_AVAILABLE (all bits set) rather
// than zero when the driver will not account it — always, under WDDM.
constexpr std::uint64_t kMemoryNotAvailable = ~0ULL;

// The sentinel must not reach the wire. A consumer reading the raw value as a
// number sees 18 exabytes of GPU memory; memory_reported is the only field
// that carries the truth.
RawProcess make_raw(unsigned int pid, unsigned long long used) {
    const bool reported = static_cast<std::uint64_t>(used) != kMemoryNotAvailable;
    return {pid, reported ? static_cast<std::uint64_t>(used) : 0, reported};
}

// The pid may have exited between NVML reporting it and this read. The memory
// figure is still true as of snapshot time, so the row is kept with a
// placeholder rather than dropped.
std::string read_process_name(std::uint32_t pid) {
    std::ifstream comm("/proc/" + std::to_string(pid) + "/comm");
    std::string name;
    if (comm && std::getline(comm, name) && !name.empty()) {
        return name;
    }
    return "<pid " + std::to_string(pid) + ">";
}

std::int64_t now_unix_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// nvmlDeviceGetProcessUtilization takes a CPU timestamp in microseconds and
// returns only samples newer than it. The clock is the realtime epoch, the same
// one system_clock reports.
unsigned long long now_unix_us() {
    using namespace std::chrono;
    return static_cast<unsigned long long>(
        duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}

}  // namespace

struct DeviceMonitor::Impl {
    ProcessQuery compute_query;
    ProcessQuery graphics_query;
    MemoryQueryV2 memory_query_v2 = nullptr;
    std::string driver_version;

    // stderr should describe a degraded field once, not once per second.
    std::set<std::string> warned;

    void warn_once(const std::string& key, const std::string& message) {
        if (warned.insert(key).second) {
            std::fprintf(stderr, "gpuflow: %s\n", message.c_str());
        }
    }

    std::vector<RawProcess> collect_processes(nvmlDevice_t device, unsigned int index,
                                              bool& listing_supported);
    void apply_process_utilization(nvmlDevice_t device, unsigned int index,
                                   std::vector<ProcessSample>& processes);
};

std::vector<RawProcess> DeviceMonitor::Impl::collect_processes(nvmlDevice_t device,
                                                              unsigned int index,
                                                              bool& listing_supported) {
    std::vector<RawProcess> merged;
    bool any_supported = false;

    // Compute and graphics are separate lists. A process can appear in both —
    // and under WDDM the graphics list is where a game or the compositor shows
    // up, which is half the point of a shared-box view.
    const std::pair<const ProcessQuery*, const char*> sources[] = {
        {&compute_query, "compute"},
        {&graphics_query, "graphics"},
    };

    for (const auto& [query, label] : sources) {
        if (!query->available()) {
            continue;
        }

        unsigned int count = 0;
        nvmlReturn_t rc = NVML_ERROR_UNKNOWN;
        std::vector<RawProcess> batch;

        if (query->wide != nullptr) {
            std::vector<nvmlProcessInfo_v2_t> infos;
            rc = fetch_processes(query->wide, device, infos, count);
            if (rc == NVML_SUCCESS) {
                for (unsigned int i = 0; i < count; ++i) {
                    batch.push_back(make_raw(infos[i].pid, infos[i].usedGpuMemory));
                }
            }
        } else {
            std::vector<nvmlProcessInfo_v1_t> infos;
            rc = fetch_processes(query->narrow, device, infos, count);
            if (rc == NVML_SUCCESS) {
                for (unsigned int i = 0; i < count; ++i) {
                    batch.push_back(make_raw(infos[i].pid, infos[i].usedGpuMemory));
                }
            }
        }

        if (rc == NVML_SUCCESS) {
            any_supported = true;
            for (const RawProcess& proc : batch) {
                auto existing = std::find_if(
                    merged.begin(), merged.end(),
                    [&](const RawProcess& p) { return p.pid == proc.pid; });
                if (existing == merged.end()) {
                    merged.push_back(proc);
                } else if (!existing->memory_reported && proc.memory_reported) {
                    existing->used_gpu_memory_bytes = proc.used_gpu_memory_bytes;
                    existing->memory_reported = true;
                }
            }
        } else if (rc != NVML_ERROR_NOT_SUPPORTED) {
            warn_once(std::string("proclist:") + label + ":" + std::to_string(index),
                      std::string("gpu ") + std::to_string(index) + ": " + label +
                          " process list unavailable (" + nvmlErrorString(rc) + ")");
        }
    }

    listing_supported = any_supported;
    return merged;
}

void DeviceMonitor::Impl::apply_process_utilization(nvmlDevice_t device, unsigned int index,
                                                    std::vector<ProcessSample>& processes) {
    if (processes.empty()) {
        return;
    }

    // A window wider than the poll interval, so a sample landing just after the
    // previous tick is still picked up rather than falling between polls.
    constexpr unsigned long long kLookbackUs = 2'000'000;
    const unsigned long long now = now_unix_us();
    const unsigned long long since = now > kLookbackUs ? now - kLookbackUs : 0;

    std::vector<nvmlProcessUtilizationSample_t> samples(processes.size() + 8);
    unsigned int count = static_cast<unsigned int>(samples.size());
    nvmlReturn_t rc = nvmlDeviceGetProcessUtilization(device, samples.data(), &count, since);
    if (rc == NVML_ERROR_INSUFFICIENT_SIZE) {
        samples.assign(count + 8, nvmlProcessUtilizationSample_t{});
        count = static_cast<unsigned int>(samples.size());
        rc = nvmlDeviceGetProcessUtilization(device, samples.data(), &count, since);
    }

    if (rc != NVML_SUCCESS) {
        // NOT_FOUND simply means no sample landed in the window — common at
        // idle, and not worth a line on stderr.
        if (rc != NVML_ERROR_NOT_FOUND) {
            warn_once("procutil:" + std::to_string(index),
                      std::string("gpu ") + std::to_string(index) +
                          ": per-process utilization unavailable (" + nvmlErrorString(rc) +
                          ") — process rows will show memory only");
        }
        return;
    }

    // The window holds several samples per pid; the newest is the one to show.
    std::unordered_map<std::uint32_t, const nvmlProcessUtilizationSample_t*> newest;
    for (unsigned int i = 0; i < count; ++i) {
        auto [it, inserted] = newest.emplace(samples[i].pid, &samples[i]);
        if (!inserted && samples[i].timeStamp > it->second->timeStamp) {
            it->second = &samples[i];
        }
    }

    for (ProcessSample& proc : processes) {
        auto it = newest.find(proc.pid);
        if (it != newest.end()) {
            proc.sm_utilization_percent = static_cast<double>(it->second->smUtil);
        }
    }
}

DeviceMonitor::DeviceMonitor() : impl_(std::make_unique<Impl>()) {
    const nvmlReturn_t rc = nvmlInit();
    if (rc != NVML_SUCCESS) {
        throw std::runtime_error(std::string("NVML init failed: ") + nvmlErrorString(rc) +
                                 " — is the NVIDIA driver loaded? Under WSL2, confirm "
                                 "`nvidia-smi` works first.");
    }

    impl_->compute_query = resolve_process_query("nvmlDeviceGetComputeRunningProcesses");
    impl_->graphics_query = resolve_process_query("nvmlDeviceGetGraphicsRunningProcesses");
    if (void* symbol = dlsym(RTLD_DEFAULT, "nvmlDeviceGetMemoryInfo_v2")) {
        impl_->memory_query_v2 = function_cast<MemoryQueryV2>(symbol);
    }

    if (!impl_->compute_query.available() && !impl_->graphics_query.available()) {
        impl_->warn_once("proclist:none",
                         "no running-process entry point found in libnvidia-ml; "
                         "device totals only");
    }

    char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    if (nvmlSystemGetDriverVersion(driver, sizeof(driver)) == NVML_SUCCESS) {
        impl_->driver_version = driver;
    }
}

DeviceMonitor::~DeviceMonitor() {
    nvmlShutdown();
}

Snapshot DeviceMonitor::poll() {
    Snapshot snapshot;
    snapshot.timestamp_unix_ms = now_unix_ms();
    snapshot.driver_version = impl_->driver_version;

    unsigned int device_count = 0;
    const nvmlReturn_t count_rc = nvmlDeviceGetCount(&device_count);
    if (count_rc != NVML_SUCCESS) {
        impl_->warn_once("devicecount", std::string("device enumeration failed: ") +
                                            nvmlErrorString(count_rc));
        return snapshot;
    }

    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t device{};
        if (nvmlDeviceGetHandleByIndex(i, &device) != NVML_SUCCESS) {
            continue;
        }

        GpuSample gpu;
        gpu.index = i;

        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE] = {};
        if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
            gpu.name = name;
        }

        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = {};
        if (nvmlDeviceGetUUID(device, uuid, sizeof(uuid)) == NVML_SUCCESS) {
            gpu.uuid = uuid;
        }

        nvmlUtilization_t utilization{};
        if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
            gpu.utilization_percent = utilization.gpu;
            gpu.memory_utilization_percent = utilization.memory;
        }

        bool memory_read = false;
        if (impl_->memory_query_v2 != nullptr) {
            nvmlMemory_v2_t memory{};
            memory.version = NVML_STRUCT_VERSION(Memory, 2);
            if (impl_->memory_query_v2(device, &memory) == NVML_SUCCESS) {
                gpu.memory_used_bytes = memory.used;
                gpu.memory_reserved_bytes = memory.reserved;
                gpu.memory_total_bytes = memory.total;
                memory_read = true;
            }
        }
        if (!memory_read) {
            // v1 has no way to separate the reservation, so `used` here is the
            // larger figure. Left as-is rather than guessed at.
            nvmlMemory_t memory{};
            if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
                gpu.memory_used_bytes = memory.used;
                gpu.memory_total_bytes = memory.total;
            }
        }

        bool listing_supported = false;
        for (const RawProcess& raw : impl_->collect_processes(device, i, listing_supported)) {
            ProcessSample proc;
            proc.pid = raw.pid;
            proc.process_name = read_process_name(raw.pid);
            proc.used_gpu_memory_bytes = raw.used_gpu_memory_bytes;
            proc.memory_reported = raw.memory_reported;
            gpu.processes.push_back(std::move(proc));
        }
        gpu.process_listing_supported = listing_supported;

        impl_->apply_process_utilization(device, i, gpu.processes);

        std::sort(gpu.processes.begin(), gpu.processes.end(),
                  [](const ProcessSample& a, const ProcessSample& b) {
                      if (a.used_gpu_memory_bytes != b.used_gpu_memory_bytes) {
                          return a.used_gpu_memory_bytes > b.used_gpu_memory_bytes;
                      }
                      return a.pid < b.pid;
                  });

        snapshot.gpus.push_back(std::move(gpu));
    }

    return snapshot;
}

}  // namespace gpuflow
