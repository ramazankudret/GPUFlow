// The agent's half of the injection loop, standalone.
//
// Creates the ring and the name table, launches a command with the collector
// injected, drains what comes back, and prints it. This is what `gpuflow run`
// will do once the events feed Snapshot; keeping it separate first means the
// injection path can be proven against a real CUDA program before any of it is
// wired into the model or the UI.
//
// Usage: collector_probe [--capacity N] [--verbose] -- <command> [args...]

#include <sys/wait.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "ipc/name_table.hpp"
#include "ipc/shared_ring.hpp"
#include "ipc/spsc_ring.hpp"

namespace {

using gpuflow::Event;
using gpuflow::EventKind;

std::string collector_path() {
    // Next to this binary, the way the agent will find it after an install.
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "libgpuflow_inject.so";
    buf[n] = '\0';
    std::string dir(buf);
    const std::size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos) dir.resize(slash);
    return dir + "/libgpuflow_inject.so";
}

const char* kind_name(std::uint8_t k) {
    switch (static_cast<EventKind>(k)) {
        case EventKind::kKernel: return "kernel";
        case EventKind::kMemcpy: return "memcpy";
        case EventKind::kStreamCreated: return "stream+";
        case EventKind::kStreamDestroyed: return "stream-";
        default: return "?";
    }
}

std::string direction(std::uint8_t flags) {
    if (flags & gpuflow::kFlagHostToDevice) return "H2D";
    if (flags & gpuflow::kFlagDeviceToHost) return "D2H";
    if (flags & gpuflow::kFlagDeviceToDevice) return "D2D";
    return "—";
}

struct Tally {
    std::uint64_t count = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t bytes = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t capacity = 1u << 16;
    std::uint32_t name_bytes = 256 * 1024;
    bool verbose = false;
    int first = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--") { first = i + 1; break; }
        if (a == "--capacity" && i + 1 < argc) capacity = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--verbose") verbose = true;
        else { std::fprintf(stderr, "unexpected argument '%s'\n", a.c_str()); return 1; }
    }
    if (first < 0 || first >= argc) {
        std::fprintf(stderr, "usage: collector_probe [--capacity N] [--verbose] -- <command> [args...]\n");
        return 1;
    }

    const std::uint64_t pid = static_cast<std::uint64_t>(::getpid());
    const std::string ring_name = gpuflow::ring_name_for_pid(pid);
    const std::string names_name = gpuflow::name_table_name_for_pid(pid);

    gpuflow::SharedRing ring = gpuflow::SharedRing::create(ring_name, capacity);
    gpuflow::SharedRing names =
        gpuflow::SharedRing::create_raw(names_name, gpuflow::name_table_bytes(name_bytes));
    gpuflow::name_table_init(names.base(), name_bytes);

    const gpuflow::NameTableGeometry name_geo =
        gpuflow::name_table_validate(names.base(), names.mapped_bytes());
    gpuflow::NameReader name_reader(names.base(), name_geo);

    const std::string inject = collector_path();
    if (::access(inject.c_str(), R_OK) != 0) {
        std::fprintf(stderr, "collector not found at %s\n", inject.c_str());
        return 1;
    }

    std::printf("ring %s (%u records)  names %s (%u bytes)\n", ring_name.c_str(), capacity,
                names_name.c_str(), name_bytes);
    std::printf("injecting %s\n\n", inject.c_str());

    const pid_t child = ::fork();
    if (child < 0) { std::perror("fork"); return 1; }

    if (child == 0) {
        // CUDA reads CUDA_INJECTION64_PATH at initialisation, which is why
        // GPUFlow can instrument what it launches and not what is already
        // running. Setting it here, before exec, is the whole mechanism.
        ::setenv("CUDA_INJECTION64_PATH", inject.c_str(), 1);
        ::setenv(gpuflow::kRingEnvVar, ring_name.c_str(), 1);
        ::setenv(gpuflow::kNameTableEnvVar, names_name.c_str(), 1);
        if (verbose) ::setenv("GPUFLOW_VERBOSE", "1", 1);
        ::execvp(argv[first], argv + first);
        std::fprintf(stderr, "exec %s: %s\n", argv[first], std::strerror(errno));
        ::_exit(127);
    }

    gpuflow::RingConsumer consumer = ring.consumer();
    std::vector<Event> batch(512);
    std::map<std::string, Tally> by_kernel;
    Tally memcpys[3];
    std::uint64_t received = 0, torn = 0;
    std::vector<Event> first_few;
    bool child_gone = false;
    int status = 0;

    for (;;) {
        const std::size_t n = consumer.pop_batch(batch.data(), batch.size());
        if (n == 0) {
            if (child_gone) break;
            if (::waitpid(child, &status, WNOHANG) == child) { child_gone = true; continue; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        name_reader.refresh();
        for (std::size_t i = 0; i < n; ++i) {
            const Event& e = batch[i];
            if (!gpuflow::event_verify(e)) { ++torn; continue; }
            ++received;
            if (first_few.size() < 8) first_few.push_back(e);

            const std::uint64_t dur = e.end_ns > e.start_ns ? e.end_ns - e.start_ns : 0;
            if (e.kind == static_cast<std::uint8_t>(EventKind::kKernel)) {
                Tally& t = by_kernel[name_reader.resolve(e.name_id)];
                ++t.count;
                t.total_ns += dur;
            } else if (e.kind == static_cast<std::uint8_t>(EventKind::kMemcpy)) {
                const int slot = (e.flags & gpuflow::kFlagHostToDevice)   ? 0
                                 : (e.flags & gpuflow::kFlagDeviceToHost) ? 1
                                                                          : 2;
                ++memcpys[slot].count;
                memcpys[slot].total_ns += dur;
                memcpys[slot].bytes += e.value_a;
            }
        }
    }
    if (!child_gone) ::waitpid(child, &status, 0);
    name_reader.refresh();

    const gpuflow::RingStats s = consumer.stats();

    std::printf("first events seen\n");
    for (const Event& e : first_few) {
        const std::uint64_t dur = e.end_ns > e.start_ns ? e.end_ns - e.start_ns : 0;
        std::printf("  %-7s dev=%u stream=%-3u %8" PRIu64 " ns  %s\n", kind_name(e.kind),
                    e.device_id, e.stream_id, dur,
                    e.kind == static_cast<std::uint8_t>(EventKind::kKernel)
                        ? name_reader.resolve(e.name_id).c_str()
                        : direction(e.flags).c_str());
    }

    std::printf("\nkernels by name\n");
    for (const auto& [name, t] : by_kernel) {
        std::printf("  %-44s %6" PRIu64 " launches  avg %7" PRIu64 " ns\n",
                    name.empty() ? "<unnamed>" : name.c_str(), t.count,
                    t.count ? t.total_ns / t.count : 0);
    }

    std::printf("\ncopies\n");
    const char* labels[3] = {"H2D", "D2H", "other"};
    for (int i = 0; i < 3; ++i) {
        if (memcpys[i].count == 0) continue;
        std::printf("  %-5s %6" PRIu64 " copies  %10" PRIu64 " bytes  avg %6" PRIu64 " ns\n",
                    labels[i], memcpys[i].count, memcpys[i].bytes,
                    memcpys[i].total_ns / memcpys[i].count);
    }

    std::printf("\nreceived %" PRIu64 "  torn %" PRIu64 "  ring dropped %" PRIu64
                "  names %zu  refused %" PRIu64 "\n",
                received, torn, s.dropped, name_reader.size() - 1, name_reader.refused());
    std::printf("child exited %s %d\n", WIFSIGNALED(status) ? "by signal" : "with code",
                WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));

    if (received == 0) {
        std::printf("\nNO EVENTS — the collector did not attach, or CUPTI delivered nothing\n");
        return 1;
    }
    return torn == 0 ? 0 : 1;
}
