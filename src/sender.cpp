// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "sender.h"

#include <ws2tcpip.h>

#include <chrono>
#include <cstring>

#include "stadia.h"

static constexpr uint16_t RECEIVER_PORT = 42734;
// UDP has no retransmission, so we repeat the current state periodically
// to recover from lost packets (and from the receiver rebooting when it
// switches to the Stadia descriptor).
static constexpr uint64_t KEEPALIVE_MS = 100;
static constexpr int NEUTRAL_RESENDS = 3;
static constexpr uint64_t NEUTRAL_RESEND_MS = 20;

void Sender::start() {
    quit_ = false;
    thread_ = std::thread(&Sender::thread_main, this);
}

void Sender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Sender::set_receiver(const std::string& ip) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        receiver_ = ip;
        receiver_pending_ = true;
        wake_ = true;
    }
    cv_.notify_one();
}

void Sender::set_active(bool active) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active == active_) {
            return;
        }
        active_ = active;
        wake_ = true;
    }
    cv_.notify_one();
}

void Sender::set_state(const GamepadState& state) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state == state_) {
            return;
        }
        state_ = state;
        wake_ = true;
    }
    cv_.notify_one();
}

void Sender::send_packet(const GamepadState& state) {
    last_sent_ = state;
    last_send_ms_ = GetTickCount64();
    if (!have_dest_ || socket_ == INVALID_SOCKET) {
        return;
    }
    uint8_t data[STADIA_PACKET_SIZE];
    size_t len = build_stadia_packet(state, data);
    int res = sendto(socket_, (const char*) data, (int) len, 0, (const sockaddr*) &dest_, sizeof(dest_));
    send_error_ = (res == SOCKET_ERROR);
    if (res != SOCKET_ERROR) {
        packets_sent_++;
    }
}

void Sender::thread_main() {
    socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_ != INVALID_SOCKET) {
        u_long nonblocking = 1;
        ioctlsocket(socket_, FIONBIO, &nonblocking);
    }
    bool was_active = false;
    int neutral_resends = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (!quit_) {
        // Until the next keepalive (or neutral resend) is due.
        const uint64_t interval = neutral_resends > 0 ? NEUTRAL_RESEND_MS : KEEPALIVE_MS;
        const uint64_t since = GetTickCount64() - last_send_ms_;
        uint64_t wait = since < interval ? interval - since : 1;
        if (!was_active && neutral_resends == 0) {
            wait = KEEPALIVE_MS;  // nothing to send until something changes
        }
        cv_.wait_for(lock, std::chrono::milliseconds(wait), [this] { return wake_ || quit_; });
        wake_ = false;
        if (quit_) {
            break;
        }
        if (receiver_pending_) {
            receiver_pending_ = false;
            memset(&dest_, 0, sizeof(dest_));
            dest_.sin_family = AF_INET;
            dest_.sin_port = htons(RECEIVER_PORT);
            have_dest_ = inet_pton(AF_INET, receiver_.c_str(), &dest_.sin_addr) == 1;
            address_valid_ = have_dest_;
            send_error_ = false;
        }
        const bool active = active_;
        const GamepadState state = state_;
        lock.unlock();

        if (active != was_active) {
            was_active = active;
            neutral_resends = active ? 0 : NEUTRAL_RESENDS;
        }
        const uint64_t now = GetTickCount64();
        if (active) {
            if (state != last_sent_ || now - last_send_ms_ >= KEEPALIVE_MS) {
                send_packet(state);
            }
        } else if (neutral_resends > 0 && now - last_send_ms_ >= NEUTRAL_RESEND_MS) {
            send_packet(GamepadState());
            neutral_resends--;
        }
        lock.lock();
    }
    lock.unlock();
    // Leave the receiver with nothing pressed.
    if (was_active) {
        send_packet(GamepadState());
    }
    if (socket_ != INVALID_SOCKET) {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
    }
}
