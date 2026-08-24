#include "collector/trace_accumulator.hpp"

#include <cxxabi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace gpuflow {
namespace {

std::uint64_t kernel_key(std::uint32_t name_index, std::uint32_t stream_id) {
    return (static_cast<std::uint64_t>(name_index) << 32) | stream_id;
}

// A demangled C++ template can run to several hundred characters, and the name
// pool is sent whole on every frame. Long enough to identify a kernel, short
// enough that twenty of them do not become the payload.
constexpr std::size_t kMaxDisplayName = 160;

// Done on the agent side, never in the injected library: the collector's job is
// to get bytes out of someone else's process, and demangling is presentation.
std::string demangle(const std::string& raw) {
    if (raw.size() < 3 || raw[0] != '_' || raw[1] != 'Z') return raw;
    int status = 0;
    char* out = abi::__cxa_demangle(raw.c_str(), nullptr, nullptr, &status);
    if (out == nullptr) return raw;
    std::string result = status == 0 ? std::string(out) : raw;
    std::free(out);
    return result;
}

std::string shorten(const std::string& name) {
    if (name.size() <= kMaxDisplayName) return name;

    // Cut the middle, not the tail. A template instantiation's identity lives
    // at both ends, and dropping the suffix makes two different kernels read as
    // the same one.
    std::size_t head = kMaxDisplayName * 2 / 3;
    const std::size_t tail = kMaxDisplayName - head - 1;
    // Never split a UTF-8 sequence; half a code point is not valid JSON.
    while (head > 0 && (static_cast<unsigned char>(name[head]) & 0xC0) == 0x80) --head;
    return name.substr(0, head) + "…" + name.substr(name.size() - tail);
}

}  // namespace

TraceAccumulator::TraceAccumulator(std::uint32_t pid, std::string command)
    : pid_(pid), command_(std::move(command)) {
    name_pool_.emplace_back();  // index 0 is the unnamed kernel
    name_index_.emplace(0u, 0u);
}

std::uint32_t TraceAccumulator::name_index_for(std::uint32_t name_id) {
    auto it = name_index_.find(name_id);
    if (it != name_index_.end()) return it->second;

    // The collector may have published the name after this batch was written,
    // so re-read the table before giving up on an id.
    names_.refresh();
    const std::string& resolved = names_.resolve(name_id);
    if (resolved.empty()) return 0;

    const auto index = static_cast<std::uint32_t>(name_pool_.size());
    name_pool_.push_back(shorten(demangle(resolved)));
    name_index_.emplace(name_id, index);
    return index;
}

void TraceAccumulator::ingest(const Event* events, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const Event& e = events[i];
        if (!event_verify(e)) continue;  // torn or corrupt; the ring's problem, not ours
        ++events_total_;

        const auto kind = static_cast<EventKind>(e.kind);

        if (kind == EventKind::kClockSync) {
            // end_ns is the collector's CLOCK_REALTIME, start_ns is CUPTI's.
            // On Linux these agree; keeping the offset means a platform where
            // they do not still draws kernels in the right place.
            clock_offset_ns_ = static_cast<std::int64_t>(e.start_ns) -
                               static_cast<std::int64_t>(e.end_ns);
            have_clock_offset_ = true;
            continue;
        }

        const std::uint64_t duration =
            e.end_ns > e.start_ns ? e.end_ns - e.start_ns : 0;

        if (kind == EventKind::kMemcpy) {
            CopyStat* stat = &d2d_;
            std::uint8_t span_kind = 3;
            if (e.flags & kFlagHostToDevice) { stat = &h2d_; span_kind = 1; }
            else if (e.flags & kFlagDeviceToHost) { stat = &d2h_; span_kind = 2; }
            ++stat->count;
            stat->bytes += e.value_a;
            stat->total_ns += duration;
            // Copies get lanes too. A program that dedicates a stream to
            // transfers has no kernels on it, so leaving copies out of the
            // spans would erase that lane from the plane entirely.
            record_span(e.stream_id, 0, e.start_ns, duration, span_kind);
            continue;
        }

        if (kind != EventKind::kKernel) continue;

        const std::uint32_t name_index = name_index_for(e.name_id);

        KernelStat& stat = kernels_[kernel_key(name_index, e.stream_id)];
        stat.name_index = name_index;
        stat.stream_id = e.stream_id;
        ++stat.launches;
        stat.total_ns += duration;
        stat.max_ns = std::max<std::uint64_t>(stat.max_ns, duration);

        record_span(e.stream_id, name_index, e.start_ns, duration, 0);
    }
}

// Spans are capped per stream and most-recent-wins. Dropping the oldest is
// right for a live view: the lane should show what is happening now, not the
// first thing that happened after the last frame.
void TraceAccumulator::record_span(std::uint32_t stream_id, std::uint32_t name_index,
                                   std::uint64_t start_ns, std::uint64_t duration_ns,
                                   std::uint8_t kind) {
    std::deque<Span>& lane = spans_[stream_id];
    if (lane.size() >= kMaxSpansPerStream) {
        lane.pop_front();
        ++elided_[stream_id];
    }
    lane.push_back(Span{stream_id, name_index,
                        static_cast<std::int64_t>(
                            (static_cast<std::int64_t>(start_ns) - clock_offset_ns_) / 1000),
                        static_cast<std::uint32_t>(std::min<std::uint64_t>(duration_ns, ~0u)),
                        kind});
}

TracedProcess TraceAccumulator::build(std::int64_t now_unix_ms) {
    TracedProcess out;
    out.pid = pid_;
    out.command = command_;
    out.running = running_;
    out.names = name_pool_;
    out.events_total = events_total_;
    out.events_dropped = events_dropped_;
    out.clock_offset_ns = have_clock_offset_ ? clock_offset_ns_ : 0;
    out.host_to_device = h2d_;
    out.device_to_host = d2h_;
    out.device_to_device = d2d_;

    out.kernels.reserve(kernels_.size());
    for (const auto& [key, stat] : kernels_) {
        (void)key;
        out.kernels.push_back(stat);
    }
    // Heaviest first: on a shared box the question is always what is eating the
    // card, and the answer should be the first row.
    std::sort(out.kernels.begin(), out.kernels.end(),
              [](const KernelStat& a, const KernelStat& b) {
                  if (a.total_ns != b.total_ns) return a.total_ns > b.total_ns;
                  return a.launches > b.launches;
              });

    const std::int64_t now_us = now_unix_ms * 1000;
    const std::int64_t cutoff_us = now_us - kSpanWindowUs;

    if (::getenv("GPUFLOW_TRACE_DEBUG") != nullptr) {
        std::int64_t newest = 0;
        for (const auto& [st, lane] : spans_) {
            (void)st;
            if (!lane.empty()) newest = std::max(newest, lane.back().start_unix_us);
        }
        std::fprintf(stderr,
                     "[trace] now_us=%lld newest_us=%lld delta_ms=%.1f offset_ns=%lld\n",
                     (long long)now_us, (long long)newest,
                     (double)(newest - now_us) / 1000.0, (long long)clock_offset_ns_);
    }

    std::vector<std::uint32_t> streams;
    streams.reserve(spans_.size());
    for (const auto& [stream, lane] : spans_) {
        (void)lane;
        streams.push_back(stream);
    }
    std::sort(streams.begin(), streams.end());

    // A process with more streams than the plane can draw gets the lowest ids,
    // which are the ones CUDA hands out first and so the ones a reader
    // recognises. The rest are counted as elided rather than silently missing.
    if (streams.size() > kMaxStreams) {
        for (std::size_t i = kMaxStreams; i < streams.size(); ++i) {
            out.spans_elided += spans_[streams[i]].size();
        }
        streams.resize(kMaxStreams);
    }
    out.streams = streams;

    for (const std::uint32_t stream : streams) {
        std::deque<Span>& lane = spans_[stream];
        // Expire anything older than the window here rather than on ingest, so
        // a quiet stream still shows its last activity until it really ages out.
        while (!lane.empty() && lane.front().start_unix_us < cutoff_us) {
            lane.pop_front();
        }
        for (const Span& s : lane) {
            KernelSpan span;
            span.stream_id = s.stream_id;
            span.name_index = s.name_index;
            span.start_offset_us = s.start_unix_us - now_us;
            span.duration_ns = s.duration_ns;
            span.kind = s.kind;
            out.spans.push_back(span);
        }
    }

    for (const auto& [stream, count] : elided_) {
        (void)stream;
        out.spans_elided += count;
    }

    return out;
}

}  // namespace gpuflow
