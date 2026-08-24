// The passive layer: everything GPUFlow can see without touching the observed
// process. NVML gives device-wide utilization and a per-process memory figure,
// and nothing finer — no kernels, no streams, no copies. That ceiling is the
// reason v0.2 exists.
//
// nvml.h stays inside device_monitor.cpp. The pimpl below is what enforces it:
// callers get a Snapshot and never learn which library produced it.

#pragma once

#include <memory>

#include "model/snapshot.hpp"

namespace gpuflow {

class DeviceMonitor {
public:
    // Throws std::runtime_error if NVML will not initialize — no driver, no
    // permission, no GPU. That is fatal at startup; a transient failure during
    // a later poll is not, and degrades a single field instead.
    DeviceMonitor();
    ~DeviceMonitor();

    DeviceMonitor(const DeviceMonitor&) = delete;
    DeviceMonitor& operator=(const DeviceMonitor&) = delete;

    Snapshot poll();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gpuflow
