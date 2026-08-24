// The shared name table behind Event::name_id.
//
// A kernel name is a variable-length string; an Event is a fixed 64 bytes and
// the collector may not allocate per record. Both are satisfied by interning:
// a training loop launches the same twenty kernels a million times, so names
// are written once and referenced by index forever after.
//
// The table is append-only and never rewritten. That is what makes it safe to
// read while it is being written: the agent reads a prefix, and a prefix of an
// append-only blob is always a complete, consistent set of earlier entries.
// One atomic — published_bytes — is the entire publication protocol.
//
// Entry 0 is the empty string. name_id 0 therefore always resolves, and means
// "no name", which is what a record gets when the table filled up or the name
// was missing. There is no id that fails to resolve.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpuflow {

constexpr std::uint32_t kNameTableMagic = 0x474e414d;  // 'GNAM'
constexpr std::uint32_t kNameTableVersion = 1;
constexpr std::size_t kNameTableDataOffset = 4096;

// A kernel name is a mangled C++ symbol; long ones run to a few hundred bytes.
constexpr std::uint32_t kMaxNameLength = 1023;

struct NameTableControl {
    std::atomic<std::uint32_t> magic;
    std::atomic<std::uint32_t> version;
    std::atomic<std::uint32_t> capacity_bytes;

    // The single publication point. Bytes below this offset are complete and
    // will never change; bytes at or above it do not exist yet.
    std::atomic<std::uint32_t> published_bytes;

    // Names the collector wanted to intern after the blob filled. Surfaced so
    // "some kernels show as unnamed" is a reported condition rather than a
    // mystery.
    std::atomic<std::uint64_t> refused;
};

static_assert(sizeof(NameTableControl) <= kNameTableDataOffset,
              "name table control block must fit before the data page");

constexpr std::size_t name_table_bytes(std::uint32_t capacity_bytes) noexcept {
    return kNameTableDataOffset + capacity_bytes;
}

inline void name_table_init(void* base, std::uint32_t capacity_bytes) noexcept {
    std::memset(base, 0, kNameTableDataOffset);
    auto* c = ::new (base) NameTableControl();
    c->capacity_bytes.store(capacity_bytes, std::memory_order_relaxed);
    c->refused.store(0, std::memory_order_relaxed);
    c->version.store(kNameTableVersion, std::memory_order_relaxed);
    // Entry 0: the empty string, so name_id 0 always resolves.
    static_cast<char*>(base)[kNameTableDataOffset] = '\0';
    c->published_bytes.store(1, std::memory_order_release);
    c->magic.store(kNameTableMagic, std::memory_order_release);
}

struct NameTableGeometry {
    std::uint32_t capacity_bytes = 0;
    bool ok = false;
    explicit operator bool() const noexcept { return ok; }
};

// Same discipline as the ring: the geometry is copied out once, and nothing
// downstream reads it again from a page the other process can write.
inline NameTableGeometry name_table_validate(const void* base, std::size_t mapped_bytes) noexcept {
    NameTableGeometry geo;
    if (base == nullptr || mapped_bytes <= kNameTableDataOffset) return geo;
    const auto* c = static_cast<const NameTableControl*>(base);
    if (c->magic.load(std::memory_order_acquire) != kNameTableMagic) return geo;
    if (c->version.load(std::memory_order_relaxed) != kNameTableVersion) return geo;

    const std::uint32_t capacity = c->capacity_bytes.load(std::memory_order_relaxed);
    if (capacity == 0) return geo;
    if (name_table_bytes(capacity) > mapped_bytes) return geo;

    geo.capacity_bytes = capacity;
    geo.ok = true;
    return geo;
}

// Collector side. Interning is the only place this code allocates, and it does
// so once per distinct kernel name, never per launch.
class NameWriter {
public:
    NameWriter() noexcept = default;
    NameWriter(void* base, const NameTableGeometry& geo) noexcept
        : c_(geo.ok ? static_cast<NameTableControl*>(base) : nullptr),
          blob_(geo.ok ? static_cast<char*>(base) + kNameTableDataOffset : nullptr),
          capacity_(geo.capacity_bytes),
          used_(1) {}  // entry 0 is already there

    bool valid() const noexcept { return c_ != nullptr; }

    // Returns 0 (the empty name) rather than failing, so a full table degrades
    // to unnamed kernels instead of dropping records.
    std::uint32_t intern(const char* name) noexcept {
        if (c_ == nullptr || name == nullptr || *name == '\0') return 0;

        std::size_t len = ::strnlen(name, kMaxNameLength + 1);
        if (len > kMaxNameLength) len = kMaxNameLength;
        const std::string key(name, len);

        auto it = ids_.find(key);
        if (it != ids_.end()) return it->second;

        if (used_ + len + 1 > capacity_) {
            c_->refused.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        std::memcpy(blob_ + used_, key.data(), len);
        blob_[used_ + len] = '\0';
        const std::uint32_t id = next_id_;

        // Bytes first, then the offset that publishes them. A reader that
        // acquires published_bytes sees a complete string or nothing.
        c_->published_bytes.store(static_cast<std::uint32_t>(used_ + len + 1),
                                  std::memory_order_release);

        used_ += len + 1;
        ++next_id_;
        ids_.emplace(key, id);
        return id;
    }

private:
    NameTableControl* c_ = nullptr;
    char* blob_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 1;
    std::uint32_t next_id_ = 1;
    std::unordered_map<std::string, std::uint32_t> ids_;
};

// Agent side. Re-reads only the part of the blob it has not parsed yet, so
// polling it every tick costs nothing once the program's kernels are known.
class NameReader {
public:
    NameReader() noexcept = default;
    NameReader(const void* base, const NameTableGeometry& geo) noexcept
        : c_(geo.ok ? static_cast<const NameTableControl*>(base) : nullptr),
          blob_(geo.ok ? static_cast<const char*>(base) + kNameTableDataOffset : nullptr),
          capacity_(geo.capacity_bytes) {
        names_.emplace_back();  // entry 0
        parsed_ = 1;
    }

    bool valid() const noexcept { return c_ != nullptr; }

    // Call before resolving ids from a fresh batch of records.
    void refresh() noexcept {
        if (c_ == nullptr) return;
        std::size_t published = c_->published_bytes.load(std::memory_order_acquire);
        if (published > capacity_) published = capacity_;  // never trust it past the mapping

        while (parsed_ < published) {
            const char* start = blob_ + parsed_;
            const std::size_t room = published - parsed_;
            const std::size_t len = ::strnlen(start, room);
            if (len == room) break;  // no terminator yet: a partial tail, wait
            names_.emplace_back(start, len);
            parsed_ += len + 1;
        }
    }

    // Every id resolves; an id past the end reads as unnamed rather than as a
    // bounds error, because the id came from another process.
    const std::string& resolve(std::uint32_t id) const noexcept {
        return id < names_.size() ? names_[id] : names_[0];
    }

    std::size_t size() const noexcept { return names_.size(); }
    std::uint64_t refused() const noexcept {
        return c_ ? c_->refused.load(std::memory_order_relaxed) : 0;
    }

private:
    const NameTableControl* c_ = nullptr;
    const char* blob_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t parsed_ = 0;
    std::vector<std::string> names_;
};

}  // namespace gpuflow
