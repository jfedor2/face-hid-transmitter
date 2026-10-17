// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <winsock2.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "gamepad.h"

// Sends the gamepad state to the receiver over UDP on its own thread: on every
// change, and every 100 ms because UDP has no retransmission. When inactive,
// sends a few all-released reports and then nothing.
class Sender {
public:
    void start();
    void stop();

    void set_receiver(const std::string& ip);
    void set_active(bool active);
    void set_state(const GamepadState& state);

    bool address_valid() const { return address_valid_.load(); }
    bool send_error() const { return send_error_.load(); }
    uint64_t packets_sent() const { return packets_sent_.load(); }

private:
    void thread_main();
    void send_packet(const GamepadState& state);

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool quit_ = false;
    bool wake_ = false;
    GamepadState state_;
    bool active_ = false;
    std::string receiver_;
    bool receiver_pending_ = false;

    std::atomic<bool> address_valid_{ false };
    std::atomic<bool> send_error_{ false };
    std::atomic<uint64_t> packets_sent_{ 0 };

    // Sender thread only.
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_in dest_{};
    bool have_dest_ = false;
    GamepadState last_sent_;
    uint64_t last_send_ms_ = 0;
};
