// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "tracker.h"

#include <chrono>
#include <cmath>
#include <cstring>

// After losing the face, the last output is held this long (it's often lost
// for a frame or two, e.g. when a hand passes in front), then released.
static constexpr double HOLD_SECONDS = 0.3;
// One Euro filter speed coefficient (per degree/second).
static constexpr float FILTER_BETA = 0.05f;

static double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Smoothing 0..1 -> minimum cutoff frequency, 10 Hz (barely any) to 0.3 Hz.
static float min_cutoff(float smoothing) {
    return 10.0f * std::pow(0.03f, smoothing);
}

bool HeadTracker::start(Sender* sender, std::string& error) {
    sender_ = sender;
    if (!face_.init(error)) {
        return false;
    }
    quit_ = false;
    thread_ = std::thread(&HeadTracker::thread_main, this);
    return true;
}

void HeadTracker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void HeadTracker::push_frame(std::shared_ptr<const Image> frame) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = std::move(frame);
    }
    cv_.notify_one();
}

void HeadTracker::set_settings(const TrackerSettings& settings) {
    std::lock_guard<std::mutex> lock(mutex_);
    settings_ = settings;
    settings_changed_ = true;
}

std::shared_ptr<const TrackerSnapshot> HeadTracker::snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

void HeadTracker::thread_main() {
    OneEuroFilter filters[HEAD_AXIS_COUNT];
    TrackerSettings settings;
    uint64_t seq = 0;
    double last_frame_time = 0;
    double last_seen = -1e9;
    float fps = 0;
    GamepadState held;
    float held_angles[HEAD_AXIS_COUNT] = {};

    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        // Wake up now and then without frames too, so a camera that stopped
        // still releases the output.
        cv_.wait_for(lock, std::chrono::milliseconds(100), [this] { return pending_ != nullptr || quit_; });
        if (quit_) {
            break;
        }
        std::shared_ptr<const Image> frame = std::move(pending_);
        pending_ = nullptr;
        if (settings_changed_) {
            settings_changed_ = false;
            // A new neutral would make the filtered angles jump.
            if (std::memcmp(&settings.neutral, &settings_.neutral, sizeof(Mat3)) != 0) {
                for (OneEuroFilter& f : filters) {
                    f.reset();
                }
            }
            settings = settings_;
        }
        lock.unlock();

        const double now = now_seconds();
        auto snap = std::make_shared<TrackerSnapshot>();
        if (frame) {
            if (last_frame_time > 0) {
                const float instant = (float) (1.0 / std::max(now - last_frame_time, 1e-3));
                fps = fps == 0 ? instant : fps + (instant - fps) * 0.1f;
            }
            last_frame_time = now;
            const double t0 = now_seconds();
            snap->face = face_.process(*frame);
            snap->process_ms = (float) ((now_seconds() - t0) * 1000);
            snap->frame = frame;
        } else {
            if (now - last_frame_time > 1.0) {
                fps = 0;
            }
            std::shared_ptr<const TrackerSnapshot> prev = snapshot();
            if (prev) {
                // keep showing the last frame
                snap->frame = prev->frame;
                snap->face = prev->face;
            }
        }

        if (frame && snap->face.found) {
            if (now - last_seen > HOLD_SECONDS) {
                for (OneEuroFilter& f : filters) {
                    f.reset();
                }
            }
            last_seen = now;
            float raw[HEAD_AXIS_COUNT];
            head_angles(snap->face.rotation, settings.neutral, raw[0], raw[1], raw[2]);
            GamepadState state;
            for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
                snap->angles[a] = filters[a].filter(raw[a], now, min_cutoff(settings.smoothing), FILTER_BETA);
                held_angles[a] = snap->angles[a];
                apply_axis(settings.axes[a], axis_output(settings.axes[a], snap->angles[a]), state);
            }
            clamp_state(state);
            held = state;
        } else if (now - last_seen > HOLD_SECONDS) {
            held = GamepadState();
        } else {
            for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
                snap->angles[a] = held_angles[a];
            }
        }
        snap->state = held;
        snap->fps = fps;
        snap->seq = ++seq;
        if (sender_) {
            sender_->set_state(held);
        }

        lock.lock();
        snapshot_ = std::move(snap);
    }
}
