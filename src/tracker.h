// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "face.h"
#include "mapping.h"
#include "sender.h"

struct TrackerSettings {
    AxisMapping axes[HEAD_AXIS_COUNT];
    Mat3 neutral;
    float smoothing = 0.5f;  // 0..1
};

// The result of processing one camera frame.
struct TrackerSnapshot {
    std::shared_ptr<const Image> frame;
    FaceResult face;
    // Head angles relative to neutral, smoothed, in degrees (when face.found).
    float angles[HEAD_AXIS_COUNT] = {};
    GamepadState state;  // what's being output
    uint64_t seq = 0;
    float fps = 0.0f;         // camera frames per second
    float process_ms = 0.0f;  // face tracking time per frame
};

// Runs the face tracker on the latest camera frame on its own thread (frames
// that arrive while it's busy are skipped), turns the head angles into
// gamepad state and hands it to the sender.
class HeadTracker {
public:
    bool start(Sender* sender, std::string& error);
    void stop();

    // Called from the camera thread.
    void push_frame(std::shared_ptr<const Image> frame);
    void set_settings(const TrackerSettings& settings);

    std::shared_ptr<const TrackerSnapshot> snapshot();

private:
    void thread_main();

    Sender* sender_ = nullptr;
    FaceTracker face_;
    std::thread thread_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool quit_ = false;
    std::shared_ptr<const Image> pending_;
    TrackerSettings settings_;
    bool settings_changed_ = false;
    std::shared_ptr<const TrackerSnapshot> snapshot_;
};
