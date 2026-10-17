// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "config.h"

#include <windows.h>

#include <knownfolders.h>
#include <shlobj.h>

#include <algorithm>
#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

Config::Config() {
    axes[(int) HeadAxis::Yaw] = { AxisTarget::LeftStickX, false, 25.0f, 4.0f };
    axes[(int) HeadAxis::Pitch] = { AxisTarget::LeftStickY, false, 15.0f, 3.0f };
    axes[(int) HeadAxis::Roll] = { AxisTarget::None, false, 20.0f, 4.0f };
}

// The same layout SDL_GetPrefPath() gives the other HID Forwarder tools.
std::wstring app_data_dir() {
    static std::wstring dir = []() {
        PWSTR appdata = nullptr;
        if (FAILED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, KF_FLAG_CREATE, nullptr, &appdata))) {
            CoTaskMemFree(appdata);
            return std::wstring(L".");
        }
        std::wstring path = appdata;
        CoTaskMemFree(appdata);
        for (const wchar_t* part : { L"\\face-hid-transmitter", L"\\face-hid-transmitter" }) {
            path += part;
            CreateDirectoryW(path.c_str(), nullptr);
        }
        return path;
    }();
    return dir;
}

template <typename E, int N>
static bool enum_from_name(const std::string& name, const char* const (&names)[N], E& out) {
    for (int i = 0; i < N; i++) {
        if (name == names[i]) {
            out = (E) i;
            return true;
        }
    }
    return false;
}

bool load_config(const std::wstring& path, Config& config) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    try {
        json j = json::parse(f);
        config.receiver_ip = j.value("receiver_ip", "");
        config.camera_id = j.value("camera_id", "");
        config.privacy = j.value("privacy", false);
        config.zoom_to_face = j.value("zoom_to_face", true);
        config.mirror = j.value("mirror", true);
        config.smoothing = std::clamp(j.value("smoothing", 0.5f), 0.0f, 1.0f);
        if (j.contains("axes") && j["axes"].is_object()) {
            for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
                const json& ja = j["axes"].value(HEAD_AXIS_NAMES[a], json::object());
                AxisMapping& m = config.axes[a];
                enum_from_name(ja.value("output", ""), AXIS_TARGET_NAMES, m.target);
                m.invert = ja.value("invert", m.invert);
                m.range = std::clamp(ja.value("range", m.range), 2.0f, 90.0f);
                m.deadzone = std::clamp(ja.value("deadzone", m.deadzone), 0.0f, 30.0f);
            }
        }
        if (j.contains("neutral") && j["neutral"].is_array() && j["neutral"].size() == 9) {
            Mat3 n;
            bool ok = true;
            for (int i = 0; i < 9; i++) {
                ok &= j["neutral"][i].is_number();
                if (ok) {
                    n.m[i / 3][i % 3] = j["neutral"][i].get<float>();
                }
            }
            // must be a rotation
            const Mat3 nn = n * n.transposed();
            for (int r = 0; r < 3 && ok; r++) {
                for (int c = 0; c < 3; c++) {
                    ok &= std::abs(nn.m[r][c] - (r == c ? 1.0f : 0.0f)) < 1e-3f;
                }
            }
            if (ok) {
                config.neutral = n;
                config.neutral_set = true;
            }
        }
        if (j.contains("window") && j["window"].is_object()) {
            const json& jw = j["window"];
            config.window.x = jw.value("x", 0);
            config.window.y = jw.value("y", 0);
            config.window.w = jw.value("w", 0);
            config.window.h = jw.value("h", 0);
            config.window.maximized = jw.value("maximized", false);
            config.window.valid = config.window.w > 0 && config.window.h > 0;
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool save_config(const std::wstring& path, const Config& config) {
    json j;
    j["receiver_ip"] = config.receiver_ip;
    j["camera_id"] = config.camera_id;
    j["privacy"] = config.privacy;
    j["zoom_to_face"] = config.zoom_to_face;
    j["mirror"] = config.mirror;
    j["smoothing"] = config.smoothing;
    json axes = json::object();
    for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
        const AxisMapping& m = config.axes[a];
        axes[HEAD_AXIS_NAMES[a]] = {
            { "output", AXIS_TARGET_NAMES[(int) m.target] },
            { "invert", m.invert },
            { "range", m.range },
            { "deadzone", m.deadzone },
        };
    }
    j["axes"] = axes;
    if (config.neutral_set) {
        json n = json::array();
        for (int i = 0; i < 9; i++) {
            n.push_back(config.neutral.m[i / 3][i % 3]);
        }
        j["neutral"] = n;
    }
    if (config.window.valid) {
        j["window"] = {
            { "x", config.window.x },
            { "y", config.window.y },
            { "w", config.window.w },
            { "h", config.window.h },
            { "maximized", config.window.maximized },
        };
    }

    std::wstring tmp_path = path + L".tmp";
    {
        std::ofstream f(tmp_path, std::ios::trunc);
        if (!f) {
            return false;
        }
        f << j.dump(2);
        if (!f) {
            return false;
        }
    }
    return MoveFileExW(tmp_path.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
}
