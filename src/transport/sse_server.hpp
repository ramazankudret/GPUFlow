// HTTP + Server-Sent Events over hand-rolled POSIX sockets.
//
// SSE rather than WebSocket: telemetry flows one way, and EventSource
// reconnects on its own with no client code. When v0.4 needs UI-to-agent
// control messages, add a route alongside this one — the dispatch and the
// client registry are kept generic for that, but nothing here speaks
// WebSocket today.
//
// Single-threaded poll(2) reactor. A handful of local browser tabs does not
// justify threads, and staying single-threaded means the NVML poll and the
// broadcast share a timeline with no locking between them.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace gpuflow {

class SseServer {
public:
    struct Options {
        std::string bind_address = "127.0.0.1";
        std::uint16_t port = 7717;
        // Path to the UI document, resolved by the caller. Read once at
        // startup so a running agent does not depend on the file surviving.
        std::string ui_path;
        int poll_interval_ms = 1000;
    };

    // Produces the payload broadcast on each tick and sent to a client the
    // moment it connects. Returning the snapshot as a string keeps the
    // transport ignorant of the model.
    using SnapshotSource = std::function<std::string()>;

    SseServer(Options options, SnapshotSource source);
    ~SseServer();

    SseServer(const SseServer&) = delete;
    SseServer& operator=(const SseServer&) = delete;

    // Throws std::runtime_error if the listening socket cannot be opened.
    void start();

    // Runs until stop() is called from a signal handler. Blocks.
    void run();

    // Async-signal-safe: sets a flag the loop checks.
    void stop();

    std::uint16_t port() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gpuflow
