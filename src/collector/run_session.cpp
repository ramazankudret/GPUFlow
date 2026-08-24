#include "collector/run_session.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "ipc/name_table.hpp"

namespace gpuflow {
namespace {

// Bounded so a process producing a hundred thousand events a second cannot
// starve the socket loop. Whatever is left stays in the ring for the next tick,
// which is what the ring is for.
constexpr std::size_t kDrainBudget = 32768;
constexpr std::uint32_t kNameTableBytes = 256 * 1024;

std::string join_command(const std::vector<std::string>& command) {
    std::string out;
    for (std::size_t i = 0; i < command.size(); ++i) {
        if (i > 0) out += ' ';
        out += command[i];
    }
    return out;
}

}  // namespace

std::string default_collector_path() {
    char buffer[4096];
    const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) return "libgpuflow_inject.so";
    buffer[length] = '\0';

    std::string path(buffer);
    const std::size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
    return dir + "/libgpuflow_inject.so";
}

RunSession::RunSession(const std::vector<std::string>& command, std::string collector_path,
                       std::uint32_t ring_capacity)
    : batch_(4096) {
    if (command.empty()) throw std::runtime_error("run needs a command");

    if (::access(collector_path.c_str(), R_OK) != 0) {
        throw std::runtime_error("collector library not found at " + collector_path +
                                 " — build with CUPTI available, or this build has no collector");
    }

    const auto agent_pid = static_cast<std::uint64_t>(::getpid());
    const std::string ring_name = ring_name_for_pid(agent_pid);
    const std::string names_name = name_table_name_for_pid(agent_pid);

    ring_ = SharedRing::create(ring_name, ring_capacity);
    names_ = SharedRing::create_raw(names_name, name_table_bytes(kNameTableBytes));
    name_table_init(names_.base(), kNameTableBytes);
    consumer_ = ring_.consumer();

    // Fork after both segments exist and are initialised: the child attaches
    // during CUDA init, which can happen before the parent's next instruction.
    const ::pid_t child = ::fork();
    if (child < 0) throw std::runtime_error(std::string("fork: ") + std::strerror(errno));

    if (child == 0) {
        ::setenv("CUDA_INJECTION64_PATH", collector_path.c_str(), 1);
        ::setenv(kRingEnvVar, ring_name.c_str(), 1);
        ::setenv(kNameTableEnvVar, names_name.c_str(), 1);
#ifdef GPUFLOW_CUPTI_LIBRARY
        // Without this, a program that names its streams through NVTX gets
        // those names dropped on the floor and the lanes stay numbered. Costs
        // nothing to a program that never calls NVTX. Not overwritten: a user
        // who set it deliberately knows something we do not.
        ::setenv("NVTX_INJECTION64_PATH", GPUFLOW_CUPTI_LIBRARY, 0);
#endif

        std::vector<char*> argv;
        argv.reserve(command.size() + 1);
        for (const std::string& arg : command) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);

        ::execvp(argv[0], argv.data());
        std::fprintf(stderr, "gpuflow: exec %s: %s\n", argv[0], std::strerror(errno));
        ::_exit(127);
    }

    child_ = child;
    running_ = true;

    accumulator_ = std::make_unique<TraceAccumulator>(static_cast<std::uint32_t>(child),
                                                      join_command(command));
    const NameTableGeometry geometry = name_table_validate(names_.base(), names_.mapped_bytes());
    accumulator_->set_name_reader(NameReader(names_.base(), geometry));
    accumulator_->set_running(true);
}

RunSession::~RunSession() {
    terminate_child();
}

void RunSession::drain() {
    std::size_t taken = 0;
    while (taken < kDrainBudget) {
        const std::size_t n = consumer_.pop_batch(batch_.data(), batch_.size());
        if (n == 0) break;
        accumulator_->ingest(batch_.data(), n);
        taken += n;
    }
    accumulator_->set_events_dropped(consumer_.stats().dropped);

    if (running_) {
        int status = 0;
        const ::pid_t reaped = ::waitpid(child_, &status, WNOHANG);
        if (reaped == child_) {
            // One more sweep before declaring it gone: the collector's exit
            // flush can publish after the process is already unwaitable.
            const std::size_t n = consumer_.pop_batch(batch_.data(), batch_.size());
            if (n > 0) accumulator_->ingest(batch_.data(), n);
            accumulator_->set_events_dropped(consumer_.stats().dropped);

            running_ = false;
            status_ = status;
            accumulator_->set_running(false);
        }
    }
}

TracedProcess RunSession::build(std::int64_t now_unix_ms) {
    return accumulator_->build(now_unix_ms);
}

void RunSession::terminate_child() {
    if (!running_ || child_ <= 0) return;
    // A child GPUFlow launched should not outlive the agent that launched it;
    // the operator would have no way left to see or stop it.
    ::kill(child_, SIGTERM);
    running_ = false;
}

}  // namespace gpuflow
