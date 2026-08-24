// One `gpuflow run` session: the segments, the child process, and the drain.
//
// This is the piece that makes the injection mechanism a feature rather than a
// proof. It owns both shared segments — the agent creates and unlinks them
// because it outlives the child, and a collector killed with its host would
// otherwise leave them behind — launches the command with the collector in
// CUDA_INJECTION64_PATH, and empties the ring on the agent's fast timer.
//
// The constraint this file exists to respect: CUDA reads CUDA_INJECTION64_PATH
// at initialisation, so GPUFlow can instrument what it launches and nothing
// that is already running. That is why `run` is a subcommand and not a flag on
// `watch`.

#pragma once

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "collector/trace_accumulator.hpp"
#include "ipc/event.hpp"
#include "ipc/shared_ring.hpp"
#include "model/snapshot.hpp"

namespace gpuflow {

class RunSession {
public:
    // Throws std::runtime_error if the segments cannot be created, the
    // collector library is missing, or the fork fails.
    RunSession(const std::vector<std::string>& command, std::string collector_path,
               std::uint32_t ring_capacity);
    ~RunSession();

    RunSession(const RunSession&) = delete;
    RunSession& operator=(const RunSession&) = delete;

    // Empties the ring into the accumulator and reaps the child if it has
    // exited. Bounded work per call so one busy process cannot stall the
    // server's poll loop.
    void drain();

    TracedProcess build(std::int64_t now_unix_ms);

    bool child_running() const { return running_; }
    int exit_status() const { return status_; }
    ::pid_t pid() const { return child_; }

    // Sent when the agent is shutting down: a child launched by GPUFlow should
    // not outlive it.
    void terminate_child();

private:
    SharedRing ring_;
    SharedRing names_;
    RingConsumer consumer_;
    std::unique_ptr<TraceAccumulator> accumulator_;

    ::pid_t child_ = -1;
    bool running_ = false;
    int status_ = 0;
    std::vector<Event> batch_;
};

// Where the injected library sits relative to the running binary. Exposed so
// `run` can fail with a useful message before forking rather than leaving the
// child to fail silently at CUDA init.
std::string default_collector_path();

}  // namespace gpuflow
