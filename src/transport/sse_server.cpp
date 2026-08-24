#include "transport/sse_server.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpuflow {
namespace {

enum class ClientKind {
    kHttpRequest,  // still reading a request line; route not yet decided
    kSseStream,    // upgraded to a long-lived event stream
};

struct Client {
    int fd = -1;
    ClientKind kind = ClientKind::kHttpRequest;
    std::string inbox;
    std::string outbox;
    bool close_after_flush = false;
};

// A request line and a few headers. Anything larger from a local browser is a
// client we do not want to keep buffering for.
constexpr std::size_t kMaxRequestBytes = 8192;

// A browser that has stopped reading its event stream is gone, whatever its
// socket still claims. Dropping it beats growing a buffer forever.
constexpr std::size_t kMaxOutboxBytes = 1u << 20;

bool set_non_blocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags != -1 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot read UI document at " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string http_response(const char* status, const char* content_type,
                          const std::string& body) {
    std::string out;
    out.reserve(body.size() + 160);
    out += "HTTP/1.1 ";
    out += status;
    out += "\r\nContent-Type: ";
    out += content_type;
    out += "\r\nContent-Length: ";
    out += std::to_string(body.size());
    out += "\r\nConnection: close\r\n\r\n";
    out += body;
    return out;
}

// Only the request line matters — this server answers GET and nothing else, so
// a full header parser would be code with no reader.
std::string request_target(const std::string& request) {
    const std::size_t line_end = request.find("\r\n");
    const std::string line = request.substr(0, line_end);

    const std::size_t first_space = line.find(' ');
    if (first_space == std::string::npos) return {};
    const std::size_t second_space = line.find(' ', first_space + 1);
    if (second_space == std::string::npos) return {};

    if (line.compare(0, first_space, "GET") != 0) return {};

    std::string target = line.substr(first_space + 1, second_space - first_space - 1);
    const std::size_t query = target.find('?');
    if (query != std::string::npos) {
        target.resize(query);
    }
    return target;
}

std::int64_t steady_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

struct SseServer::Impl {
    Options options;
    SnapshotSource source;

    int listen_fd = -1;
    std::uint16_t bound_port = 0;
    std::string ui_document;
    std::vector<Client> clients;

    // poll() has to wake for a signal as well as for socket activity, and a
    // handler cannot safely do more than write a byte.
    int wake_read = -1;
    int wake_write = -1;
    std::atomic<bool> running{false};

    ~Impl() {
        for (Client& client : clients) {
            if (client.fd >= 0) ::close(client.fd);
        }
        if (listen_fd >= 0) ::close(listen_fd);
        if (wake_read >= 0) ::close(wake_read);
        if (wake_write >= 0) ::close(wake_write);
    }

    void open_listener();
    void accept_pending();
    void handle_readable(Client& client);
    void route(Client& client, const std::string& target);
    void flush(Client& client);
    void queue(Client& client, const std::string& payload);
    bool has_stream_clients() const;
    void broadcast(const std::string& json);
};

void SseServer::Impl::open_listener() {
    listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
    }

    const int reuse = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    if (::inet_pton(AF_INET, options.bind_address.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("invalid bind address: " + options.bind_address);
    }

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        throw std::runtime_error("bind " + options.bind_address + ":" +
                                 std::to_string(options.port) + ": " + std::strerror(errno));
    }
    if (::listen(listen_fd, 16) != 0) {
        throw std::runtime_error(std::string("listen: ") + std::strerror(errno));
    }
    if (!set_non_blocking(listen_fd)) {
        throw std::runtime_error(std::string("fcntl: ") + std::strerror(errno));
    }

    sockaddr_in actual{};
    socklen_t actual_len = sizeof(actual);
    if (::getsockname(listen_fd, reinterpret_cast<sockaddr*>(&actual), &actual_len) == 0) {
        bound_port = ntohs(actual.sin_port);
    } else {
        bound_port = options.port;
    }
}

void SseServer::Impl::accept_pending() {
    for (;;) {
        const int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            break;  // EAGAIN once the backlog is drained
        }
        if (!set_non_blocking(fd)) {
            ::close(fd);
            continue;
        }
        Client client;
        client.fd = fd;
        clients.push_back(std::move(client));
    }
}

void SseServer::Impl::queue(Client& client, const std::string& payload) {
    if (client.outbox.size() + payload.size() > kMaxOutboxBytes) {
        client.close_after_flush = true;
        client.outbox.clear();
        return;
    }
    client.outbox += payload;
    flush(client);
}

void SseServer::Impl::flush(Client& client) {
    while (!client.outbox.empty()) {
        // MSG_NOSIGNAL, not a SIGPIPE handler: a client vanishing mid-write is
        // routine and must not be able to signal the process at all.
        const ssize_t written =
            ::send(client.fd, client.outbox.data(), client.outbox.size(), MSG_NOSIGNAL);
        if (written > 0) {
            client.outbox.erase(0, static_cast<std::size_t>(written));
            continue;
        }
        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;  // retried on the next POLLOUT
        }
        client.close_after_flush = true;
        client.outbox.clear();
        return;
    }
}

void SseServer::Impl::route(Client& client, const std::string& target) {
    if (target == "/") {
        queue(client, http_response("200 OK", "text/html; charset=utf-8", ui_document));
        client.close_after_flush = true;
        return;
    }

    if (target == "/events") {
        client.kind = ClientKind::kSseStream;
        std::string headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "X-Accel-Buffering: no\r\n"
            "\r\n";
        // A tab that just reloaded should paint immediately rather than sit
        // blank until the next tick.
        headers += "data: " + source() + "\n\n";
        queue(client, headers);
        return;
    }

    queue(client, http_response("404 Not Found", "text/plain; charset=utf-8", "not found\n"));
    client.close_after_flush = true;
}

void SseServer::Impl::handle_readable(Client& client) {
    char buffer[2048];
    for (;;) {
        const ssize_t received = ::recv(client.fd, buffer, sizeof(buffer), 0);
        if (received > 0) {
            if (client.kind == ClientKind::kSseStream) {
                continue;  // nothing a client says on an event stream matters
            }
            client.inbox.append(buffer, static_cast<std::size_t>(received));
            if (client.inbox.size() > kMaxRequestBytes) {
                client.close_after_flush = true;
                return;
            }
            const std::size_t end = client.inbox.find("\r\n\r\n");
            if (end != std::string::npos) {
                const std::string target = request_target(client.inbox);
                if (target.empty()) {
                    queue(client, http_response("405 Method Not Allowed",
                                                "text/plain; charset=utf-8", "GET only\n"));
                    client.close_after_flush = true;
                } else {
                    route(client, target);
                }
                return;
            }
            continue;
        }

        if (received == 0) {
            client.close_after_flush = true;  // tab closed or navigated away
            client.outbox.clear();
            return;
        }

        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;
        }
        client.close_after_flush = true;
        client.outbox.clear();
        return;
    }
}

bool SseServer::Impl::has_stream_clients() const {
    for (const Client& client : clients) {
        if (client.kind == ClientKind::kSseStream && !client.close_after_flush) {
            return true;
        }
    }
    return false;
}

void SseServer::Impl::broadcast(const std::string& json) {
    const std::string frame = "data: " + json + "\n\n";
    for (Client& client : clients) {
        if (client.kind == ClientKind::kSseStream && !client.close_after_flush) {
            queue(client, frame);
        }
    }
}

SseServer::SseServer(Options options, SnapshotSource source)
    : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
    impl_->source = std::move(source);
}

SseServer::~SseServer() = default;

std::uint16_t SseServer::port() const {
    return impl_->bound_port;
}

void SseServer::start() {
    impl_->ui_document = read_file(impl_->options.ui_path);

    int wake[2] = {-1, -1};
    if (::pipe(wake) != 0) {
        throw std::runtime_error(std::string("pipe: ") + std::strerror(errno));
    }
    set_non_blocking(wake[0]);
    set_non_blocking(wake[1]);
    impl_->wake_read = wake[0];
    impl_->wake_write = wake[1];

    impl_->open_listener();
}

void SseServer::stop() {
    impl_->running = false;
    if (impl_->wake_write >= 0) {
        const char byte = 'x';
        ssize_t ignored = ::write(impl_->wake_write, &byte, 1);
        (void)ignored;
    }
}

void SseServer::run() {
    impl_->running = true;

    std::int64_t next_tick_ms = steady_now_ms();
    std::vector<pollfd> fds;

    while (impl_->running) {
        fds.clear();
        fds.push_back({impl_->listen_fd, POLLIN, 0});
        fds.push_back({impl_->wake_read, POLLIN, 0});
        for (const Client& client : impl_->clients) {
            short events = POLLIN;
            if (!client.outbox.empty()) events |= POLLOUT;
            fds.push_back({client.fd, events, 0});
        }

        const std::int64_t now = steady_now_ms();
        const int timeout = static_cast<int>(next_tick_ms > now ? next_tick_ms - now : 0);

        const int ready = ::poll(fds.data(), fds.size(), timeout);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }

        if (fds[0].revents & POLLIN) {
            impl_->accept_pending();
        }

        if (fds[1].revents & POLLIN) {
            char drain[64];
            while (::read(impl_->wake_read, drain, sizeof(drain)) > 0) {
            }
        }

        // The client vector may have grown in accept_pending(); indices from
        // this poll only describe the clients that existed when it was built.
        const std::size_t polled_clients = fds.size() - 2;
        for (std::size_t i = 0; i < polled_clients && i < impl_->clients.size(); ++i) {
            Client& client = impl_->clients[i];
            const short revents = fds[i + 2].revents;
            if (revents == 0) continue;

            if (revents & POLLOUT) {
                impl_->flush(client);
            }
            if (revents & POLLIN) {
                impl_->handle_readable(client);
            }
            if (revents & (POLLERR | POLLHUP | POLLNVAL)) {
                client.close_after_flush = true;
                client.outbox.clear();
            }
        }

        if (steady_now_ms() >= next_tick_ms) {
            // An idle agent should be idle. Sampling NVML on a timer with
            // nobody watching costs a driver round trip per second forever.
            if (impl_->has_stream_clients()) {
                impl_->broadcast(impl_->source());
            }
            next_tick_ms = steady_now_ms() + impl_->options.poll_interval_ms;
        }

        for (auto it = impl_->clients.begin(); it != impl_->clients.end();) {
            if (it->close_after_flush && it->outbox.empty()) {
                ::close(it->fd);
                it = impl_->clients.erase(it);
            } else {
                ++it;
            }
        }
    }
}

}  // namespace gpuflow
