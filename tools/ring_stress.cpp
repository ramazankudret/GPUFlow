// Synthetic exerciser for the SPSC event ring.
//
// The ring is built and proven here, before any CUPTI code exists, because the
// alternative is debugging a lock-free protocol from inside a library injected
// into someone else's CUDA process. A missed release fence looks identical to a
// CUPTI misuse from in there.
//
// Invariants asserted on the streaming modes:
//   1. Every record read passes its own checksum      — no torn writes.
//   2. Sequences strictly increase                    — no reorder, no duplicate.
//   3. received + dropped == attempted, where attempted is what the producer
//      *reports having tried*, not a figure derived from the ring's own
//      counters. Deriving it would make the check restate itself.
//   4. Every sequence in [0, attempted) is either received or accounted for as
//      a drop. This has to include the runs missing before the first record
//      read and after the last one: with a throttled reader the producer
//      finishes long before the reader catches up, so most of the loss is a
//      trailing run rather than an interior gap.
//
// The streaming modes only ever show the ring behaving. --mode malice is the
// one that attacks it: a child that scribbles the control block instead of
// using it, to prove the consumer neither indexes outside its mapping nor
// mistakes corruption for a very full ring.
//
// Every drain loop has a deadline. A liveness property enforced by the test
// hanging is not enforced at all.

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ipc/shared_ring.hpp"
#include "ipc/spsc_ring.hpp"

namespace {

using gpuflow::Event;
using gpuflow::EventKind;
using gpuflow::RingConsumer;
using gpuflow::RingGeometry;
using gpuflow::RingProducer;

constexpr double kDeadlineSeconds = 30.0;

struct Options {
    std::string mode = "all";
    std::uint64_t records = 5'000'000;
    std::uint32_t capacity = 4096;
    std::uint32_t batch = 256;
    int reader_delay_us = 0;
};

struct Verdict {
    std::uint64_t attempted = 0;
    std::uint64_t produced = 0;
    std::uint64_t dropped = 0;
    std::uint64_t received = 0;
    std::uint64_t torn = 0;
    std::uint64_t out_of_order = 0;
    std::uint64_t gap_interior = 0;
    std::uint64_t first_seq = 0;
    std::uint64_t last_seq = 0;
    bool have_seq = false;
    bool timed_out = false;
    double seconds = 0.0;
    double producer_seconds = 0.0;

    // Records whose fate is genuinely unknowable. An asynchronous SIGKILL can
    // land between the producer publishing its intent to push record i and the
    // push itself, and no amount of bookkeeping on either side can close that
    // window — so the accounting checks admit exactly that much slack rather
    // than pretending to a precision the kernel does not offer.
    std::uint64_t slack = 0;

    std::uint64_t missing() const {
        if (!have_seq) return attempted;
        return first_seq + gap_interior + (attempted - 1 - last_seq);
    }
};

std::int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

class Deadline {
public:
    explicit Deadline(double seconds) : end_(now_ns() + static_cast<std::int64_t>(seconds * 1e9)) {}
    bool expired() const { return now_ns() > end_; }

private:
    std::int64_t end_;
};

// Page-aligned backing store. RingControl is alignas(64) and gets a placement
// new, so a vector's 16-byte alignment would be undefined behaviour — mmap
// gives page alignment on the real path, and the harness has to match it.
class RingStorage {
public:
    explicit RingStorage(std::uint32_t capacity) : bytes_(gpuflow::ring_bytes(capacity)) {
        base_ = std::aligned_alloc(gpuflow::kCacheLine,
                                   (bytes_ + gpuflow::kCacheLine - 1) / gpuflow::kCacheLine *
                                       gpuflow::kCacheLine);
        if (base_ == nullptr) {
            std::fprintf(stderr, "aligned_alloc failed\n");
            std::abort();
        }
        gpuflow::ring_init(base_, capacity);
        geometry_ = gpuflow::ring_validate(base_, bytes_);
    }
    ~RingStorage() { std::free(base_); }
    RingStorage(const RingStorage&) = delete;
    RingStorage& operator=(const RingStorage&) = delete;

    void* base() const { return base_; }
    const RingGeometry& geometry() const { return geometry_; }

private:
    void* base_ = nullptr;
    std::size_t bytes_ = 0;
    RingGeometry geometry_;
};

// Shaped like a real kernel-launch record so the throughput figure reflects the
// work the collector will actually do, not an empty struct copy.
Event make_event(std::uint64_t seq) {
    Event e{};
    e.kind = static_cast<std::uint8_t>(EventKind::kSynthetic);
    e.flags = 0;
    e.device_id = static_cast<std::uint16_t>(seq & 0x7);
    e.stream_id = static_cast<std::uint32_t>((seq >> 3) & 0xff);
    e.start_ns = seq * 1000;
    e.end_ns = seq * 1000 + 750;
    e.correlation_id = static_cast<std::uint32_t>(seq);
    e.name_id = static_cast<std::uint32_t>(seq % 20);  // a loop reuses few names
    e.value_a = 256;
    e.value_b = 128;
    e.sequence = seq;
    e.context_id = 1;
    gpuflow::event_seal(e);
    return e;
}

// One place that decides whether a record is trustworthy and where it sits in
// the sequence, so the in-process and cross-process harnesses cannot drift
// apart in what they consider a pass.
void observe(Verdict& v, const Event& e) {
    if (!gpuflow::event_verify(e)) {
        ++v.torn;
        return;
    }
    if (v.have_seq) {
        if (e.sequence <= v.last_seq) {
            // Leave last_seq alone: letting it move backwards would make every
            // later gap measurement nonsense, right when the numbers are needed.
            ++v.out_of_order;
            ++v.received;
            return;
        }
        v.gap_interior += e.sequence - v.last_seq - 1;
    } else {
        v.first_seq = e.sequence;
        v.have_seq = true;
    }
    v.last_seq = e.sequence;
    ++v.received;
}

int report(const char* label, const Verdict& v, bool expect_no_drops) {
    const double rate = v.seconds > 0 ? static_cast<double>(v.produced) / v.seconds : 0.0;

    std::printf("\n%s\n", label);
    std::printf("  attempted      %" PRIu64 "\n", v.attempted);
    std::printf("  produced       %" PRIu64 "\n", v.produced);
    std::printf("  dropped        %" PRIu64 " (%.3f%%)\n", v.dropped,
                v.attempted ? 100.0 * static_cast<double>(v.dropped) / static_cast<double>(v.attempted) : 0.0);
    std::printf("  received       %" PRIu64 "\n", v.received);
    std::printf("  torn records   %" PRIu64 "\n", v.torn);
    std::printf("  out of order   %" PRIu64 "\n", v.out_of_order);
    std::printf("  missing seqs   %" PRIu64 " (interior gaps %" PRIu64 ")\n", v.missing(),
                v.gap_interior);
    std::printf("  wall clock     %.3f s   %.2f M rec/s (includes the drain)\n", v.seconds,
                rate / 1e6);
    if (v.producer_seconds > 0 && v.attempted > 0) {
        std::printf("  producer cost  %.1f ns per try_push  (%.2f M call/s)\n",
                    v.producer_seconds * 1e9 / static_cast<double>(v.attempted),
                    static_cast<double>(v.attempted) / v.producer_seconds / 1e6);
    }

    int failures = 0;
    const auto check = [&failures](bool ok, const char* what) {
        std::printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };

    check(!v.timed_out, "drained within the deadline");
    check(v.torn == 0, "no torn records");
    check(v.out_of_order == 0, "strictly increasing sequence");
    check(v.received == v.produced, "received == produced");
    const std::uint64_t accounted = v.received + v.dropped;
    check(accounted <= v.attempted && accounted + v.slack >= v.attempted,
          v.slack ? "received + dropped == attempted (±1 for the in-flight push)"
                  : "received + dropped == attempted");
    // Independent of the counter above: proves the loss really is where the
    // ring says it is, rather than a record quietly overwritten in place.
    const std::uint64_t missing = v.missing();
    check(missing >= v.dropped && missing <= v.dropped + v.slack,
          "every missing sequence is an accounted drop");
    if (expect_no_drops) check(v.dropped == 0, "no drops when the reader keeps up");
    return failures;
}

/* ------------------------------------------------------------- in-process */

void consume_all(RingConsumer& consumer, Verdict& v, const std::atomic<bool>& producer_done,
                 const Options& opt) {
    std::vector<Event> batch(opt.batch);
    Deadline deadline(kDeadlineSeconds);

    for (;;) {
        if (deadline.expired()) {
            v.timed_out = true;
            return;
        }
        std::size_t n = consumer.pop_batch(batch.data(), batch.size());
        if (n == 0) {
            if (!producer_done.load(std::memory_order_acquire)) {
                std::this_thread::yield();
                continue;
            }
            // The producer may have published between the empty read and the
            // flag; one more sweep before calling it done.
            n = consumer.pop_batch(batch.data(), batch.size());
            if (n == 0) break;
        }

        for (std::size_t i = 0; i < n; ++i) observe(v, batch[i]);

        if (opt.reader_delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(opt.reader_delay_us));
        }
    }
}

int run_threads(const Options& opt, bool expect_no_drops) {
    RingStorage storage(opt.capacity);
    RingProducer producer(storage.base(), storage.geometry());
    RingConsumer consumer(storage.base(), storage.geometry());

    std::atomic<bool> done{false};
    Verdict v;
    v.attempted = opt.records;

    const std::int64_t t0 = now_ns();
    std::thread reader([&] { consume_all(consumer, v, done, opt); });

    const std::int64_t p0 = now_ns();
    for (std::uint64_t i = 0; i < opt.records; ++i) {
        producer.try_push(make_event(i));
    }
    v.producer_seconds = static_cast<double>(now_ns() - p0) / 1e9;

    done.store(true, std::memory_order_release);
    reader.join();
    v.seconds = static_cast<double>(now_ns() - t0) / 1e9;

    v.produced = producer.produced();
    v.dropped = producer.dropped();

    char label[160];
    std::snprintf(label, sizeof(label), "threads · capacity %u · %" PRIu64 " records%s",
                  opt.capacity, opt.records, opt.reader_delay_us ? " · throttled reader" : "");
    return report(label, v, expect_no_drops);
}

/* ------------------------------------------------------ two real processes */

// The child's own count of pushes attempted, reported through memory the ring
// does not own. Deriving `attempted` from the ring's counters would let a
// miscounted drop move the expected total to match itself.
struct ChildReport {
    std::atomic<std::uint64_t> attempted;
    std::atomic<std::uint32_t> finished;
};

ChildReport* map_child_report() {
    void* p = ::mmap(nullptr, sizeof(ChildReport), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "mmap report: %s\n", std::strerror(errno));
        std::abort();
    }
    auto* r = static_cast<ChildReport*>(p);
    r->attempted.store(0, std::memory_order_relaxed);
    r->finished.store(0, std::memory_order_relaxed);
    return r;
}

enum class ShmVariant { kClean, kKilled };

int run_shm(const Options& opt, ShmVariant variant) {
    const std::string name = gpuflow::ring_name_for_pid(static_cast<std::uint64_t>(::getpid()));
    gpuflow::SharedRing ring = gpuflow::SharedRing::create(name, opt.capacity);
    ChildReport* report_slot = map_child_report();

    const std::int64_t t0 = now_ns();
    const pid_t child = ::fork();
    if (child < 0) {
        std::fprintf(stderr, "fork: %s\n", std::strerror(errno));
        return 1;
    }

    if (child == 0) {
        // The collector's half: open an existing ring, announce, write, leave.
        gpuflow::SharedRing mine = gpuflow::SharedRing::open(name);
        mine.mark_producer_attached(static_cast<std::uint64_t>(::getpid()));
        RingProducer producer = mine.producer();
        for (std::uint64_t i = 0; i < opt.records; ++i) {
            // Intent published *before* the push. Publishing after would let a
            // kill land in between and leave the parent holding a record the
            // child never admitted to attempting, which reads as the ring
            // inventing data. This way the ambiguity points the harmless way:
            // the parent may expect one record that was never pushed.
            report_slot->attempted.store(i + 1, std::memory_order_release);
            producer.try_push(make_event(i));
        }
        report_slot->finished.store(1, std::memory_order_release);
        mine.mark_producer_detached();
        ::_exit(0);
    }

    if (variant == ShmVariant::kKilled) {
        // A real SIGKILL, delivered while the child is mid-loop rather than at
        // a tidy boundary. This is the only way to land between the record
        // memcpy and the head store — the one window where a crash could leave
        // a half-written slot, and the whole reason this mode exists.
        std::this_thread::sleep_for(std::chrono::microseconds(1500));
        ::kill(child, SIGKILL);
    }

    RingConsumer consumer = ring.consumer();
    std::vector<Event> batch(opt.batch);
    Verdict v;
    v.slack = (variant == ShmVariant::kKilled) ? 1 : 0;
    bool child_gone = false;
    int status = 0;  // must outlive the loop: this is where the child gets reaped
    Deadline deadline(kDeadlineSeconds);

    for (;;) {
        if (deadline.expired()) {
            v.timed_out = true;
            ::kill(child, SIGKILL);
            break;
        }
        const std::size_t n = consumer.pop_batch(batch.data(), batch.size());
        if (n == 0) {
            if (child_gone) break;
            // waitpid, not producer_attached: a killed collector never gets to
            // clear its own flag, so the flag can only ever be a hint.
            if (::waitpid(child, &status, WNOHANG) == child) {
                child_gone = true;  // one more drain pass, then out
                continue;
            }
            std::this_thread::yield();
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) observe(v, batch[i]);
        if (opt.reader_delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(opt.reader_delay_us));
        }
    }

    if (!child_gone) ::waitpid(child, &status, 0);
    v.seconds = static_cast<double>(now_ns() - t0) / 1e9;

    const gpuflow::RingStats s = consumer.stats();
    v.produced = s.consumed;
    v.dropped = s.dropped;
    v.attempted = report_slot->attempted.load(std::memory_order_acquire);

    char label[200];
    std::snprintf(label, sizeof(label), "shm · two processes · capacity %u%s%s", opt.capacity,
                  variant == ShmVariant::kKilled ? " · producer SIGKILLed mid-loop" : "",
                  opt.reader_delay_us ? " · throttled reader" : "");
    int failures = report(label, v, false);

    if (variant == ShmVariant::kKilled) {
        const bool died_by_signal = WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
        const bool partial = !report_slot->finished.load(std::memory_order_acquire);
        std::printf("  %s child really was killed by a signal\n", died_by_signal ? "PASS" : "FAIL");
        std::printf("  %s killed mid-stream, not at a loop boundary\n", partial ? "PASS" : "WARN");
        if (!died_by_signal) ++failures;
        std::printf("  NOTE producer_attached is still set (%d) after the kill — advisory only\n",
                    consumer.producer_attached() ? 1 : 0);
    }

    ::munmap(report_slot, sizeof(ChildReport));
    return failures;
}

/* ------------------------------------------- the trust boundary under attack */

// Everything above shows the ring working. This shows it refusing to be broken.
// Without it, S-class bugs in ring_validate and pop_batch are invisible to the
// suite by construction: a consumer that trusted a rewritten capacity would
// pass every streaming test right up until it indexed a gigabyte past its own
// mapping.
constexpr std::uint64_t kMaliceTailPushes = 32;

int run_malice(const Options& opt) {
    struct Case {
        const char* what;
        int index;
    };
    const Case cases[] = {
        {"capacity rewritten to 0 after validation", 0},
        {"capacity rewritten to a huge power of two", 1},
        {"head stored behind tail (underflows to ~2^64)", 2},
        {"head stored far ahead of anything written", 3},
    };

    int failures = 0;
    for (const Case& c : cases) {
        const std::string name =
            gpuflow::ring_name_for_pid(static_cast<std::uint64_t>(::getpid()) * 100 +
                                       static_cast<std::uint64_t>(c.index));
        gpuflow::SharedRing ring = gpuflow::SharedRing::create(name, opt.capacity);

        // Geometry captured here, before anything is corrupted. The whole point
        // is that the views below keep using this and never look again.
        RingConsumer consumer = ring.consumer();
        RingProducer producer = ring.producer();

        // Run tail well past capacity before corrupting anything. This is what
        // gives the test teeth: with tail smaller than the real capacity, a
        // widened mask still lands inside the array by luck, and a consumer
        // that trusted the rewritten value would look correct. Once tail is
        // beyond capacity, (tail + i) & a_wider_mask leaves the mapping.
        std::vector<Event> drain(opt.batch);
        for (std::uint64_t round = 0; round < 4; ++round) {
            for (std::uint64_t i = 0; i < opt.capacity; ++i) producer.try_push(make_event(i));
            while (consumer.pop_batch(drain.data(), drain.size()) != 0) {}
        }
        for (std::uint64_t i = 0; i < kMaliceTailPushes; ++i) producer.try_push(make_event(i));

        auto* ctl = static_cast<gpuflow::RingControl*>(ring.base());
        switch (c.index) {
            case 0: ctl->capacity.store(0, std::memory_order_relaxed); break;
            case 1: ctl->capacity.store(1u << 30, std::memory_order_relaxed); break;
            case 2:
                ctl->head.store(consumer.stats().consumed - 1, std::memory_order_release);
                break;
            case 3:
                ctl->head.store(consumer.stats().consumed + opt.capacity * 4ULL,
                                std::memory_order_release);
                break;
            default: break;
        }

        // Deliberately build the view AFTER the corruption. SharedRing::consumer()
        // returns by value, so an agent that writes `ring.consumer().pop_batch(...)`
        // in its poll loop constructs one of these every iteration — which is
        // precisely when a view that re-read capacity from the shared header
        // would pick up the rewritten value. Immunity has to come from the
        // geometry SharedRing captured at open, not from construction order.
        RingConsumer fresh = ring.consumer();

        // Drain hard. A consumer that re-read capacity would index outside the
        // mapping here; one that clamped an impossible count would spin forever.
        std::vector<Event>& batch = drain;
        Deadline deadline(3.0);
        std::uint64_t reads = 0, records = 0, bad = 0;
        bool spun = false;
        while (!deadline.expired()) {
            const std::size_t n = fresh.pop_batch(batch.data(), batch.size());
            // Counting records is not enough. A consumer that obeyed a widened
            // mask reads from offsets outside its own array — which, in a
            // process with plenty of other mappings, often lands somewhere
            // readable and returns quietly wrong data instead of crashing.
            // Only the contents can tell the two apart.
            for (std::size_t i = 0; i < n; ++i) {
                if (!gpuflow::event_verify(batch[i]) || batch[i].sequence >= kMaliceTailPushes) {
                    ++bad;
                }
            }
            records += n;
            if (++reads > 2'000'000) { spun = true; break; }
            if (n == 0) break;
        }

        const gpuflow::RingStats s = fresh.stats();
        const bool survived = !deadline.expired() && !spun;
        // Cases 2 and 3 are protocol-impossible states, so the ring must latch
        // a fault. Cases 0 and 1 must be *ignored entirely* — the geometry was
        // already captured, so rewriting it should change nothing at all.
        const bool wants_fault = (c.index == 2 || c.index == 3);

        std::printf("\nmalice · %s\n", c.what);
        std::printf("  reads %" PRIu64 "  records %" PRIu64 "  bogus %" PRIu64 "  faulted %s\n",
                    reads, records, bad, s.faulted ? "yes" : "no");
        const auto check = [&failures](bool ok, const char* what) {
            std::printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
            if (!ok) ++failures;
        };
        check(survived, "consumer stayed inside its mapping and did not spin");
        if (wants_fault) {
            check(s.faulted, "impossible state latched a fault instead of being clamped");
            check(fresh.pop_batch(batch.data(), batch.size()) == 0,
                  "faulted ring stays closed");
        } else {
            check(!s.faulted, "rewritten geometry changed nothing");
            check(bad == 0 && records == kMaliceTailPushes,
                  "read exactly the records that were written, from the right slots");
        }
    }
    return failures;
}

/* ----------------------------------------------------------- cost breakdown */

// The published overhead number has to say what it is measuring. The cost the
// collector imposes splits in two, and they are paid by different code:
//
//   build  — turning a CUPTI record into an Event, checksum included. Pure
//            arithmetic on registers, no sharing.
//   push   — the ring itself, which is cheap in instructions but pays for a
//            cache line the consumer is pulling away on another core.
//
// Measured with a consumer draining flat out, which is the worst case for that
// line transfer; a real agent polls in batches and contends far less.
int run_cost(const Options& opt) {
    RingStorage storage(1u << 20);
    RingProducer producer(storage.base(), storage.geometry());
    RingConsumer consumer(storage.base(), storage.geometry());
    std::atomic<bool> done{false};

    std::thread reader([&] {
        std::vector<Event> batch(opt.batch);
        while (!done.load(std::memory_order_acquire)) {
            if (consumer.pop_batch(batch.data(), batch.size()) == 0) std::this_thread::yield();
        }
        while (consumer.pop_batch(batch.data(), batch.size()) != 0) {}
    });

    const std::uint64_t n = opt.records;

    // Sink defeats dead-store elimination without adding a memory barrier.
    std::uint64_t sink = 0;
    const std::int64_t b0 = now_ns();
    for (std::uint64_t i = 0; i < n; ++i) sink += make_event(i).check;
    const double build_s = static_cast<double>(now_ns() - b0) / 1e9;

    Event prebuilt = make_event(0);
    const std::int64_t p0 = now_ns();
    for (std::uint64_t i = 0; i < n; ++i) {
        prebuilt.sequence = i;
        producer.try_push(prebuilt);
    }
    const double push_s = static_cast<double>(now_ns() - p0) / 1e9;

    done.store(true, std::memory_order_release);
    reader.join();

    const double build_ns = build_s * 1e9 / static_cast<double>(n);
    const double push_ns = push_s * 1e9 / static_cast<double>(n);

    std::printf("\ncost breakdown · %" PRIu64 " records · consumer draining flat out\n", n);
    std::printf("  build event    %6.1f ns   (activity record -> Event, checksum included)\n", build_ns);
    std::printf("  try_push       %6.1f ns   (ring, worst-case cache contention)\n", push_ns);
    std::printf("  total per event%6.1f ns\n", build_ns + push_ns);
    std::printf("  drops          %" PRIu64 "  (sink %" PRIu64 ")\n", producer.dropped(), sink & 1);
    std::printf("\n  At that cost, a program launching kernels at:\n");
    for (const std::uint64_t rate : {10'000ULL, 100'000ULL, 1'000'000ULL}) {
        const double frac = (build_ns + push_ns) * static_cast<double>(rate) / 1e9;
        std::printf("    %9" PRIu64 " /s  ->  %6.3f%% of one core\n", rate, frac * 100.0);
    }
    return 0;
}

/* ------------------------------------------------- SPSC violation detector */

// Two concurrent producers is the failure mode that would be hardest to
// diagnose in the field: no crash, no error, just records silently interleaved
// into the same slot. Two concurrent consumers hand out the same batch twice.
// GPUFLOW_RING_DEBUG turns both into an immediate abort — and an untested guard
// is not a guard, so this proves they fire.
int run_guard(const Options& opt) {
#ifndef GPUFLOW_RING_DEBUG
    (void)opt;
    std::printf("\nguard · skipped — configure with -DGPUFLOW_RING_DEBUG=ON to exercise it\n");
    return 0;
#else
    int failures = 0;
    for (int which = 0; which < 2; ++which) {
        const pid_t child = ::fork();
        if (child < 0) {
            std::fprintf(stderr, "fork: %s\n", std::strerror(errno));
            return 1;
        }
        if (child == 0) {
            RingStorage storage(opt.capacity);
            RingProducer producer(storage.base(), storage.geometry());
            RingConsumer consumer(storage.base(), storage.geometry());
            std::vector<Event> sinkbuf(64);
            auto hammer = [&] {
                for (std::uint64_t i = 0; i < 2'000'000; ++i) {
                    if (which == 0) producer.try_push(make_event(i));
                    else consumer.pop_batch(sinkbuf.data(), sinkbuf.size());
                }
            };
            std::thread a(hammer), b(hammer);
            a.join();
            b.join();
            ::_exit(0);  // reaching here means the guard never noticed
        }
        int status = 0;
        ::waitpid(child, &status, 0);
        const bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
        std::printf("\nguard · two concurrent %s\n", which == 0 ? "producers" : "consumers");
        std::printf("  %s the debug guard aborts (child %s)\n", aborted ? "PASS" : "FAIL",
                    WIFSIGNALED(status) ? "died by signal" : "exited normally");
        if (!aborted) ++failures;
    }
    return failures;
#endif
}

void usage() {
    std::fprintf(stderr,
                 "usage: ring_stress [--mode all|threads|shm|slow|crash|malice|guard|cost]\n"
                 "                   [--records N] [--capacity N] [--batch N]\n"
                 "                   [--reader-delay-us N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& f = args[i];
        const bool has = i + 1 < args.size();
        if (f == "--mode" && has) opt.mode = args[++i];
        else if (f == "--records" && has) opt.records = std::strtoull(args[++i].c_str(), nullptr, 10);
        else if (f == "--capacity" && has) opt.capacity = static_cast<std::uint32_t>(std::strtoul(args[++i].c_str(), nullptr, 10));
        else if (f == "--batch" && has) opt.batch = static_cast<std::uint32_t>(std::strtoul(args[++i].c_str(), nullptr, 10));
        else if (f == "--reader-delay-us" && has) opt.reader_delay_us = std::atoi(args[++i].c_str());
        else { usage(); return 1; }
    }
    if (!gpuflow::is_power_of_two(opt.capacity) || opt.batch == 0) { usage(); return 1; }

    std::printf("gpuflow ring_stress · Event=%zu bytes · control=%zu bytes\n",
                sizeof(Event), sizeof(gpuflow::RingControl));

    int failures = 0;
    const std::string& m = opt.mode;
    const bool all = (m == "all");

    if (all || m == "threads") {
        Options o = opt;
        o.capacity = opt.capacity < 65536 ? 65536 : opt.capacity;
        failures += run_threads(o, /*expect_no_drops=*/false);
    }
    if (all || m == "slow") {
        // A tiny ring plus a throttled reader: the overflow path, exercised
        // hard, with wrap-around happening constantly. Run it across a real
        // process boundary too — the producer's acquire on tail exists solely
        // to order slot recycling, and only a full ring puts it under pressure.
        Options o = opt;
        o.capacity = 64;
        o.records = opt.records < 200000 ? opt.records : 200000;
        o.reader_delay_us = 50;
        o.batch = 16;
        failures += run_threads(o, /*expect_no_drops=*/false);
        Options p = o;
        p.records = 40000;
        failures += run_shm(p, ShmVariant::kClean);
    }
    if (all || m == "shm") {
        failures += run_shm(opt, ShmVariant::kClean);
    }
    if (all || m == "crash") {
        failures += run_shm(opt, ShmVariant::kKilled);
    }
    if (all || m == "malice") {
        failures += run_malice(opt);
    }
    if (all || m == "guard") {
        failures += run_guard(opt);
    }
    if (all || m == "cost") {
        failures += run_cost(opt);
    }

    std::printf("\n%s — %d check(s) failed\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
