// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "face.h"
#include "mapping.h"

struct WindowRect {
    bool valid = false;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    bool maximized = false;

    bool operator==(const WindowRect& o) const {
        return valid == o.valid && x == o.x && y == o.y && w == o.w && h == o.h && maximized == o.maximized;
    }
};

struct Config {
    std::string receiver_ip;
    std::string camera_id;  // empty = first camera
    // Preview
    bool privacy = false;
    bool zoom_to_face = true;
    bool mirror = true;
    // Mapping
    AxisMapping axes[HEAD_AXIS_COUNT];
    float smoothing = 0.5f;
    bool neutral_set = false;
    Mat3 neutral;
    WindowRect window;

    Config();
};

// %APPDATA%\face-hid-transmitter\face-hid-transmitter, created if needed.
std::wstring app_data_dir();

bool load_config(const std::wstring& path, Config& config);
bool save_config(const std::wstring& path, const Config& config);
