// The agent's half of the collector: Events in, TracedProcess out.
//
// This is where the volume problem is solved. A busy program produces around a
// hundred thousand kernel records a second; a browser cannot be sent them and
// could not draw them if it were. So two representations leave here, and they
// answer different questions:
//
//   kernels — one entry per (name, stream), always complete. Its size tracks
//             how many distinct kernels a program has, not how hard it runs
//             them, so it stays small no matter the launch rate. This is what
//             answers "what is this process doing".
//   spans   — individual executions for the lanes, bounded and most-recent.
//             This is what makes the picture move, and it is necessarily a
//             sample.
//
// Because spans are sampled, spans_elided is not optional. A timeline that
// silently omits most of what happened, while looking continuous, is the exact
// dishonesty this project refuses everywhere else.

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "ipc/event.hpp"
#include "ipc/name_table.hpp"
#include "model/snapshot.hpp"

namespace gpuflow {

// Roughly one poll interval of history: enough that consecutive frames join up,
// short enough that the lanes show now rather than a minute ago.
constexpr std::int64_t kSpanWindowUs = 1'200'000;

// Per stream rather than globally, so a busy stream cannot starve a quiet one
// out of the picture entirely.
constexpr std::size_t kMaxSpansPerStream = 96;
constexpr std::size_t kMaxStreams = 24;

class TraceAccumulator {
public:
    TraceAccumulator(std::uint32_t pid, std::string command);

    void set_name_reader(NameReader reader) { names_ = std::move(reader); }
    void set_running(bool running) { running_ = running; }

    // Called from the drain hook with whatever the ring had.
    void ingest(const Event* events, std::size_t count);

    // Records the ring's own drop counter so losses inside the transport and
    // losses inside this window stay distinguishable.
    void set_events_dropped(std::uint64_t dropped) { events_dropped_ = dropped; }

    // Builds the view for one frame, relative to the given wall-clock moment.
    TracedProcess build(std::int64_t now_unix_ms);

private:
    struct Span {
        std::uint32_t stream_id;
        std::uint32_t name_index;
        std::int64_t start_unix_us;
        std::uint32_t duration_ns;
        std::uint8_t kind;
    };

    void record_span(std::uint32_t stream_id, std::uint32_t name_index, std::uint64_t start_ns,
                     std::uint64_t duration_ns, std::uint8_t kind);

    std::uint32_t name_index_for(std::uint32_t name_id);

    std::uint32_t pid_;
    std::string command_;
    bool running_ = false;

    NameReader names_;
    std::vector<std::string> name_pool_;
    std::unordered_map<std::uint32_t, std::uint32_t> name_index_;  // ring id -> pool index

    // Keyed by (name_index << 32) | stream_id.
    std::unordered_map<std::uint64_t, KernelStat> kernels_;
    std::unordered_map<std::uint32_t, std::deque<Span>> spans_;
    std::unordered_map<std::uint32_t, std::uint64_t> elided_;

    // What is known about a stream, as opposed to what ran on it. A stream can
    // be both the null stream and named, so the two facts accumulate rather
    // than overwrite.
    struct StreamFacts {
        std::uint32_t name_index = 0;
        bool is_default = false;
    };
    std::unordered_map<std::uint32_t, StreamFacts> stream_facts_;

    CopyStat h2d_, d2h_, d2d_;
    std::uint64_t events_total_ = 0;
    std::uint64_t events_dropped_ = 0;
    std::int64_t clock_offset_ns_ = 0;
    bool have_clock_offset_ = false;
};

}  // namespace gpuflow
