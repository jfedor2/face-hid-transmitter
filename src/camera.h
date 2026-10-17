// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "face.h"

struct CameraDevice {
    std::string id;    // symbolic link (UTF-8)
    std::string name;  // friendly name
};

std::vector<CameraDevice> list_cameras();

// Captures frames from a camera with Media Foundation and hands them to the
// sink as 32-bit BGRX (on a Media Foundation thread). Picks a format around
// 720p at 30 fps. A thread of its own opens the camera, and reopens it if it
// goes away, stops sending frames or fails to open (e.g. while another app is
// using it).
class Camera {
public:
    using Sink = std::function<void(std::shared_ptr<const Image> frame)>;

    void start(Sink sink);
    void stop();

    // Empty = the first camera.
    void set_device(const std::string& id);

    // Name and format of the camera being captured, or empty if none.
    std::string device_name();
    std::string format();
    std::string error();

private:
    friend class ReaderCallback;

    void thread_main();
    void run_device(const std::string& id);
    void set_error(const std::string& e);
    void set_format(const std::string& f);

    Sink sink_;
    std::thread thread_;
    std::atomic<bool> quit_{ false };
    std::atomic<bool> device_changed_{ false };

    std::mutex mutex_;
    std::string requested_id_;
    std::string device_name_;
    std::string format_;
    std::string error_;
    void* wake_event_ = nullptr;  // set by stop() and set_device()
};
