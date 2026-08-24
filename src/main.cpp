// Entry point: argument parsing, the poll timer, and signal handling.
//
// The wiring here is the whole reason the layers stay separable — main is the
// only place that knows DeviceMonitor, RunSession, and SseServer all exist, and
// it joins them through Snapshot rather than through a shared type.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "collector/run_session.hpp"
#include "model/snapshot.hpp"
#include "nvml/device_monitor.hpp"
#include "transport/sse_server.hpp"

namespace {

gpuflow::SseServer* g_server = nullptr;

void handle_signal(int) {
    if (g_server != nullptr) {
        g_server->stop();  // only sets a flag and writes one byte
    }
}

void print_usage() {
    std::fprintf(stderr,
                 "GPUFlow — live view of what every process is doing on the GPUs\n"
                 "\n"
                 "usage: gpuflow watch [options]\n"
                 "       gpuflow run [options] -- <command> [args...]\n"
                 "\n"
                 "  watch   the passive layer: every process on every device\n"
                 "  run     the same, plus the kernels and copies inside the command it\n"
                 "          launches. CUDA reads CUDA_INJECTION64_PATH at initialisation,\n"
                 "          so only a process GPUFlow starts can be instrumented.\n"
                 "\n"
                 "  --bind ADDRESS     interface to listen on (default 127.0.0.1)\n"
                 "  --port PORT        port to listen on (default 7717)\n"
                 "  --interval MS      sampling interval in milliseconds (default 1000)\n"
                 "  --ui PATH          override the UI document path\n"
                 "  --ring-capacity N  event ring size in records, power of two\n"
                 "                     (default 262144, about 16 MB)\n"
                 "\n"
                 "Binding beyond loopback exposes the PIDs and process names of every\n"
                 "user on this machine. See the README before using --bind 0.0.0.0.\n");
}

// The UI document is found relative to the binary, not the working directory,
// so `./build/gpuflow watch` behaves the same from anywhere in the tree.
std::string default_ui_path() {
    char buffer[4096];
    const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) {
        return "ui/index.html";
    }
    buffer[length] = '\0';

    std::string exe_path(buffer);
    const std::size_t slash = exe_path.find_last_of('/');
    const std::string exe_dir = slash == std::string::npos ? "." : exe_path.substr(0, slash);

    // Built in-tree the binary sits in build/, one level under the repo root;
    // installed alongside the UI it sits next to it.
    const std::string candidates[] = {
        exe_dir + "/../ui/index.html",
        exe_dir + "/ui/index.html",
    };
    for (const std::string& candidate : candidates) {
        if (::access(candidate.c_str(), R_OK) == 0) {
            return candidate;
        }
    }
    return candidates[0];
}

bool parse_int(const char* text, long& out) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    out = value;
    return true;
}

bool is_power_of_two(long v) {
    return v > 0 && (v & (v - 1)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help") {
        print_usage();
        return args.empty() ? 1 : 0;
    }

    const std::string subcommand = args[0];
    if (subcommand != "watch" && subcommand != "run") {
        std::fprintf(stderr, "gpuflow: unknown subcommand '%s'\n\n", subcommand.c_str());
        print_usage();
        return 1;
    }

    gpuflow::SseServer::Options options;
    options.ui_path = default_ui_path();
    std::uint32_t ring_capacity = 1u << 18;
    std::vector<std::string> command;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& flag = args[i];
        const bool has_value = i + 1 < args.size();

        // Everything past the separator belongs to the launched program, flags
        // included. Without it, a --verbose meant for the child would be
        // rejected here as an unknown gpuflow option.
        if (flag == "--") {
            command.assign(args.begin() + static_cast<long>(i) + 1, args.end());
            break;
        }

        if (flag == "--bind" && has_value) {
            options.bind_address = args[++i];
        } else if (flag == "--port" && has_value) {
            long value = 0;
            if (!parse_int(args[++i].c_str(), value) || value < 0 || value > 65535) {
                std::fprintf(stderr, "gpuflow: --port expects 0-65535\n");
                return 1;
            }
            options.port = static_cast<std::uint16_t>(value);
        } else if (flag == "--interval" && has_value) {
            long value = 0;
            if (!parse_int(args[++i].c_str(), value) || value < 50) {
                std::fprintf(stderr, "gpuflow: --interval expects at least 50 ms\n");
                return 1;
            }
            options.poll_interval_ms = static_cast<int>(value);
        } else if (flag == "--ui" && has_value) {
            options.ui_path = args[++i];
        } else if (flag == "--ring-capacity" && has_value) {
            long value = 0;
            if (!parse_int(args[++i].c_str(), value) || !is_power_of_two(value)) {
                std::fprintf(stderr, "gpuflow: --ring-capacity expects a power of two\n");
                return 1;
            }
            ring_capacity = static_cast<std::uint32_t>(value);
        } else if (subcommand == "run" && flag.rfind('-', 0) != 0) {
            // A bare word under run starts the command, so the common case
            // `gpuflow run ./train` works without needing the separator.
            command.assign(args.begin() + static_cast<long>(i), args.end());
            break;
        } else {
            std::fprintf(stderr, "gpuflow: unexpected argument '%s'\n\n", flag.c_str());
            print_usage();
            return 1;
        }
    }

    if (subcommand == "run" && command.empty()) {
        std::fprintf(stderr, "gpuflow: run needs a command\n\n");
        print_usage();
        return 1;
    }

    try {
        gpuflow::DeviceMonitor monitor;

        std::unique_ptr<gpuflow::RunSession> session;
        if (subcommand == "run") {
            session = std::make_unique<gpuflow::RunSession>(
                command, gpuflow::default_collector_path(), ring_capacity);
        }

        gpuflow::SseServer server(options, [&monitor, &session]() {
            gpuflow::Snapshot snapshot = monitor.poll();
            if (session) {
                snapshot.traced.push_back(session->build(snapshot.timestamp_unix_ms));
            }
            return snapshot.to_json();
        });

        if (session) {
            server.set_drain_hook([&session]() { session->drain(); });
        }
        server.start();

        g_server = &server;
        struct sigaction action {};
        action.sa_handler = handle_signal;
        sigemptyset(&action.sa_mask);
        ::sigaction(SIGINT, &action, nullptr);
        ::sigaction(SIGTERM, &action, nullptr);

        std::fprintf(stderr, "GPUFlow listening on http://%s:%u\n",
                     options.bind_address.c_str(), static_cast<unsigned>(server.port()));
        if (session) {
            std::fprintf(stderr,
                         "instrumenting pid %d — the plane keeps serving after it exits\n",
                         static_cast<int>(session->pid()));
        }
        if (options.bind_address != "127.0.0.1" && options.bind_address != "localhost") {
            std::fprintf(stderr,
                         "gpuflow: bound beyond loopback — this stream carries the PIDs "
                         "and process names of every user on this machine.\n");
        }

        server.run();
        g_server = nullptr;

        if (session && !session->child_running()) {
            const int status = session->exit_status();
            std::fprintf(stderr, "child exited %s %d\n",
                         WIFSIGNALED(status) ? "by signal" : "with code",
                         WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
        }
        std::fprintf(stderr, "GPUFlow stopped.\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "gpuflow: %s\n", error.what());
        return 1;
    }
}
