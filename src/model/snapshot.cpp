// Snapshot's JSON wire format, written by hand.
//
// A JSON library would earn its build friction if the schema were open-ended.
// This one is three fixed structs. Revisit past v0.3, when the kernel and
// stream layers have settled and the payload is no longer something one file
// can hold in view.

#include "model/snapshot.hpp"

#include <array>
#include <charconv>
#include <cmath>

namespace gpuflow {
namespace {

// Process names come from /proc and are effectively arbitrary bytes chosen by
// another user on the box. Everything the JSON grammar reserves has to be
// escaped here or a crafted process name breaks the stream for every client.
void append_escaped(std::string& out, const std::string& value) {
    out += '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[(c >> 4) & 0xf];
                    out += kHex[c & 0xf];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

template <typename T>
void append_number(std::string& out, T value) {
    std::array<char, 32> buf{};
    auto result = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    out.append(buf.data(), result.ptr);
}

// std::to_chars rather than snprintf: the shortest round-trip representation
// without dragging the C locale into the wire format, where a comma decimal
// separator would silently produce invalid JSON.
void append_double(std::string& out, double value) {
    if (!std::isfinite(value)) {
        out += "null";
        return;
    }
    std::array<char, 32> buf{};
    auto result = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    out.append(buf.data(), result.ptr);
}

void append_field(std::string& out, const char* key) {
    out += '"';
    out += key;
    out += "\":";
}

}  // namespace

std::string Snapshot::to_json() const {
    std::string out;
    // One SSE frame per poll on a machine with a handful of processes lands
    // well under this; the reserve just keeps the tick allocation-free.
    out.reserve(1024);

    out += '{';
    append_field(out, "timestamp_ms");
    append_number(out, timestamp_unix_ms);

    out += ',';
    append_field(out, "driver_version");
    append_escaped(out, driver_version);

    out += ',';
    append_field(out, "gpus");
    out += '[';
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        const GpuSample& gpu = gpus[i];
        if (i > 0) out += ',';

        out += '{';
        append_field(out, "index");
        append_number(out, gpu.index);
        out += ',';
        append_field(out, "name");
        append_escaped(out, gpu.name);
        out += ',';
        append_field(out, "uuid");
        append_escaped(out, gpu.uuid);
        out += ',';
        append_field(out, "utilization_percent");
        append_number(out, gpu.utilization_percent);
        out += ',';
        append_field(out, "memory_utilization_percent");
        append_number(out, gpu.memory_utilization_percent);
        out += ',';
        append_field(out, "memory_used_bytes");
        append_number(out, gpu.memory_used_bytes);
        out += ',';
        append_field(out, "memory_reserved_bytes");
        append_number(out, gpu.memory_reserved_bytes);
        out += ',';
        append_field(out, "memory_total_bytes");
        append_number(out, gpu.memory_total_bytes);

        out += ',';
        append_field(out, "process_listing_supported");
        out += gpu.process_listing_supported ? "true" : "false";

        out += ',';
        append_field(out, "processes");
        out += '[';
        for (std::size_t j = 0; j < gpu.processes.size(); ++j) {
            const ProcessSample& proc = gpu.processes[j];
            if (j > 0) out += ',';

            out += '{';
            append_field(out, "pid");
            append_number(out, proc.pid);
            out += ',';
            append_field(out, "name");
            append_escaped(out, proc.process_name);
            out += ',';
            append_field(out, "used_gpu_memory_bytes");
            append_number(out, proc.used_gpu_memory_bytes);
            out += ',';
            append_field(out, "memory_reported");
            out += proc.memory_reported ? "true" : "false";
            out += ',';
            append_field(out, "sm_utilization_percent");
            append_double(out, proc.sm_utilization_percent);
            out += '}';
        }
        out += ']';
        out += '}';
    }
    out += ']';

    out += ',';
    append_field(out, "traced");
    out += '[';
    for (std::size_t i = 0; i < traced.size(); ++i) {
        const TracedProcess& t = traced[i];
        if (i > 0) out += ',';

        out += '{';
        append_field(out, "pid");
        append_number(out, t.pid);
        out += ',';
        append_field(out, "command");
        append_escaped(out, t.command);
        out += ',';
        append_field(out, "running");
        out += t.running ? "true" : "false";
        out += ',';
        append_field(out, "events_total");
        append_number(out, t.events_total);
        out += ',';
        append_field(out, "events_dropped");
        append_number(out, t.events_dropped);
        out += ',';
        append_field(out, "spans_elided");
        append_number(out, t.spans_elided);
        out += ',';
        append_field(out, "clock_offset_ns");
        append_number(out, t.clock_offset_ns);

        out += ',';
        append_field(out, "names");
        out += '[';
        for (std::size_t j = 0; j < t.names.size(); ++j) {
            if (j > 0) out += ',';
            append_escaped(out, t.names[j]);
        }
        out += ']';

        out += ',';
        append_field(out, "streams");
        out += '[';
        for (std::size_t j = 0; j < t.streams.size(); ++j) {
            const StreamInfo& st = t.streams[j];
            if (j > 0) out += ',';
            out += '{';
            append_field(out, "id");
            append_number(out, st.id);
            out += ',';
            append_field(out, "name");
            append_number(out, st.name_index);
            out += ',';
            append_field(out, "default");
            out += st.is_default ? "true" : "false";
            out += '}';
        }
        out += ']';

        out += ',';
        append_field(out, "kernels");
        out += '[';
        for (std::size_t j = 0; j < t.kernels.size(); ++j) {
            const KernelStat& k = t.kernels[j];
            if (j > 0) out += ',';
            out += '{';
            append_field(out, "name");
            append_number(out, k.name_index);
            out += ',';
            append_field(out, "stream");
            append_number(out, k.stream_id);
            out += ',';
            append_field(out, "launches");
            append_number(out, k.launches);
            out += ',';
            append_field(out, "total_ns");
            append_number(out, k.total_ns);
            out += ',';
            append_field(out, "max_ns");
            append_number(out, k.max_ns);
            out += '}';
        }
        out += ']';

        // Positional rather than keyed: at a thousand-plus spans a frame, field
        // names would be most of the payload. [stream, name, startUs, durNs, kind].
        out += ',';
        append_field(out, "spans");
        out += '[';
        for (std::size_t j = 0; j < t.spans.size(); ++j) {
            const KernelSpan& s = t.spans[j];
            if (j > 0) out += ',';
            out += '[';
            append_number(out, s.stream_id);
            out += ',';
            append_number(out, s.name_index);
            out += ',';
            append_number(out, s.start_offset_us);
            out += ',';
            append_number(out, s.duration_ns);
            out += ',';
            append_number(out, s.kind);
            out += ']';
        }
        out += ']';

        out += ',';
        append_field(out, "copies");
        out += '{';
        const std::pair<const char*, const CopyStat*> copies[] = {
            {"h2d", &t.host_to_device},
            {"d2h", &t.device_to_host},
            {"d2d", &t.device_to_device},
        };
        for (std::size_t j = 0; j < 3; ++j) {
            if (j > 0) out += ',';
            append_field(out, copies[j].first);
            out += '{';
            append_field(out, "count");
            append_number(out, copies[j].second->count);
            out += ',';
            append_field(out, "bytes");
            append_number(out, copies[j].second->bytes);
            out += ',';
            append_field(out, "total_ns");
            append_number(out, copies[j].second->total_ns);
            out += '}';
        }
        out += '}';
        out += '}';
    }
    out += ']';
    out += '}';

    return out;
}

}  // namespace gpuflow
