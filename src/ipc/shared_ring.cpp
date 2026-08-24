#include "ipc/shared_ring.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace gpuflow {
namespace {

// errno must be captured before the message is built: the concatenations below
// allocate, and malloc is allowed to set errno even when it succeeds. Reporting
// a stale errno at exactly the moment the caller needs the real one has cost
// more debugging hours than most bugs.
[[noreturn]] void fail(const char* what, const std::string& name, int err) {
    throw std::runtime_error(std::string(what) + " '" + name + "': " + std::strerror(err));
}

}  // namespace

std::string ring_name_for_pid(std::uint64_t pid) {
    return "/gpuflow." + std::to_string(pid);
}

std::string name_table_name_for_pid(std::uint64_t pid) {
    return "/gpuflow." + std::to_string(pid) + ".names";
}

namespace {

// Shared by the ring and raw paths: map an existing segment at its own size.
void* map_existing(const std::string& name, std::size_t& bytes, std::size_t minimum) {
    const int fd = ::shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) fail("shm_open", name, errno);

    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        const int err = errno;
        ::close(fd);
        fail("fstat", name, err);
    }

    bytes = static_cast<std::size_t>(info.st_size);
    if (bytes < minimum) {
        ::close(fd);
        throw std::runtime_error("segment '" + name + "' is smaller than its own header");
    }

    // Note: the segment can still be ftruncate'd smaller by anyone running as
    // this user, after which touching a page past the new end raises SIGBUS
    // rather than a recoverable fault. Out of scope — the same-uid attacker who
    // can do that can also ptrace the agent — but it is a real property of
    // mapping a file another process controls, not an oversight.
    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) fail("mmap", name, errno);
    return base;
}

}  // namespace

SharedRing::~SharedRing() {
    if (base_ != nullptr) {
        ::munmap(base_, bytes_);
    }
    if (owns_ && !name_.empty()) {
        ::shm_unlink(name_.c_str());
    }
}

SharedRing::SharedRing(SharedRing&& other) noexcept
    : base_(other.base_),
      bytes_(other.bytes_),
      geometry_(other.geometry_),
      name_(std::move(other.name_)),
      owns_(other.owns_) {
    other.reset();
}

SharedRing& SharedRing::operator=(SharedRing&& other) noexcept {
    if (this != &other) {
        if (base_ != nullptr) ::munmap(base_, bytes_);
        if (owns_ && !name_.empty()) ::shm_unlink(name_.c_str());
        base_ = other.base_;
        bytes_ = other.bytes_;
        geometry_ = other.geometry_;
        name_ = std::move(other.name_);
        owns_ = other.owns_;
        other.reset();
    }
    return *this;
}

void SharedRing::reset() noexcept {
    base_ = nullptr;
    bytes_ = 0;
    geometry_ = RingGeometry{};
    name_.clear();
    owns_ = false;
}

SharedRing SharedRing::create(const std::string& name, std::uint32_t capacity) {
    if (!is_power_of_two(capacity) || capacity > kMaxRingCapacity) {
        throw std::runtime_error("ring capacity must be a power of two, at most " +
                                 std::to_string(kMaxRingCapacity));
    }

    // O_EXCL and no unlink-first. Removing the name before creating it would
    // defeat the exclusivity it asks for: two agents racing would each "win"
    // their O_EXCL after unlinking the other's live segment, and each would
    // then hand its child a name pointing at the other's ring. Failing on
    // EEXIST is the honest outcome — a leftover segment means either another
    // agent is running or one died, and adopting it blind would resume from a
    // stale head and tail.
    const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        const int err = errno;
        if (err == EEXIST) {
            throw std::runtime_error("ring '" + name +
                                     "' already exists — another agent is using it, or a "
                                     "previous run died; remove /dev/shm" + name +
                                     " if you are sure it is stale");
        }
        fail("shm_open", name, err);
    }

    const std::size_t bytes = ring_bytes(capacity);
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        const int err = errno;
        ::close(fd);
        ::shm_unlink(name.c_str());
        fail("ftruncate", name, err);
    }

    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);  // the mapping keeps the segment alive; the descriptor is done
    if (base == MAP_FAILED) {
        const int err = errno;
        ::shm_unlink(name.c_str());
        fail("mmap", name, err);
    }

    ring_init(base, capacity);

    SharedRing ring;
    // Ownership fields first: if the name copy throws, the destructor still has
    // what it needs to unlink, rather than leaving an orphan in /dev/shm.
    ring.name_ = name;
    ring.owns_ = true;
    ring.base_ = base;
    ring.bytes_ = bytes;
    ring.geometry_ = ring_validate(base, bytes);
    return ring;
}

SharedRing SharedRing::open(const std::string& name) {
    std::size_t bytes = 0;
    void* base = map_existing(name, bytes, kRingDataOffset + sizeof(Event));

    // The trust boundary. Everything downstream indexes off this copy of the
    // geometry, never off the header it came from.
    const RingGeometry geometry = ring_validate(base, bytes);
    if (!geometry) {
        ::munmap(base, bytes);
        throw std::runtime_error("ring '" + name + "' has no usable header");
    }

    SharedRing ring;
    ring.name_ = name;
    ring.owns_ = false;
    ring.base_ = base;
    ring.bytes_ = bytes;
    ring.geometry_ = geometry;
    return ring;
}

SharedRing SharedRing::create_raw(const std::string& name, std::size_t bytes) {
    const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        const int err = errno;
        if (err == EEXIST) {
            throw std::runtime_error("segment '" + name + "' already exists");
        }
        fail("shm_open", name, err);
    }
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        const int err = errno;
        ::close(fd);
        ::shm_unlink(name.c_str());
        fail("ftruncate", name, err);
    }
    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        const int err = errno;
        ::shm_unlink(name.c_str());
        fail("mmap", name, err);
    }

    SharedRing seg;
    seg.name_ = name;
    seg.owns_ = true;
    seg.base_ = base;
    seg.bytes_ = bytes;
    return seg;  // geometry_ stays empty: this segment is not a ring
}

SharedRing SharedRing::open_raw(const std::string& name) {
    std::size_t bytes = 0;
    void* base = map_existing(name, bytes, 1);

    SharedRing seg;
    seg.name_ = name;
    seg.owns_ = false;
    seg.base_ = base;
    seg.bytes_ = bytes;
    return seg;
}

void SharedRing::mark_producer_attached(std::uint64_t pid) noexcept {
    if (base_ == nullptr) return;
    auto* c = static_cast<RingControl*>(base_);
    c->producer_pid.store(pid, std::memory_order_relaxed);
    c->producer_attached.store(1, std::memory_order_release);
}

void SharedRing::mark_producer_detached() noexcept {
    if (base_ == nullptr) return;
    auto* c = static_cast<RingControl*>(base_);
    c->producer_attached.store(0, std::memory_order_release);
}

}  // namespace gpuflow
