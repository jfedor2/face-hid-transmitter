// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

// Command line tool for testing the face tracking without the UI or a camera.
//
//   fht-test model detect|landmark input.f32 output.f32
//       Runs one model on a raw float32 input tensor and writes its outputs
//       (concatenated, raw float32). For comparing with another TFLite runtime.
//   fht-test image photo.jpg [landmarks.csv]
//       Runs the tracker on an image file and prints what it found, the head
//       angles and the timings. Optionally writes the landmarks as x,y,z lines.
//   fht-test pose
//       Checks the head pose fit on synthetic, rotated copies of the canonical face.
//   fht-test bench
//       Times the models.

#include <windows.h>

#include <objbase.h>
#include <shellapi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "face.h"
#include "resources.h"
#include "tflite.h"

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool load_model(const std::string& which, TfliteModel& model) {
    const int id = which == "detect" ? IDR_FACE_DETECTION : IDR_FACE_LANDMARK;
    HRSRC res = FindResourceW(nullptr, MAKEINTRESOURCEW(id), (LPCWSTR) RT_RCDATA);
    HGLOBAL h = res ? LoadResource(nullptr, res) : nullptr;
    if (!h || !model.load((const uint8_t*) LockResource(h), SizeofResource(nullptr, res))) {
        fprintf(stderr, "loading %s failed: %s\n", which.c_str(), model.error().c_str());
        return false;
    }
    return true;
}

static int cmd_model(const std::string& which, const char* in_path, const char* out_path) {
    TfliteModel model;
    if (!load_model(which, model)) {
        return 1;
    }
    size_t n = 1;
    for (int d : model.input_shape()) {
        n *= d;
    }
    std::ifstream in(in_path, std::ios::binary);
    in.read((char*) model.input(), (std::streamsize) (n * sizeof(float)));
    if (!in) {
        fprintf(stderr, "input must be %zu float32 values\n", n);
        return 1;
    }
    model.run();
    std::ofstream out(out_path, std::ios::binary);
    for (size_t i = 0; i < model.output_count(); i++) {
        out.write((const char*) model.output(i), (std::streamsize) (model.output_size(i) * sizeof(float)));
    }
    return 0;
}

static int cmd_image(const wchar_t* path, const wchar_t* csv) {
    Image img;
    if (!load_image_file(path, img)) {
        fprintf(stderr, "can't load the image\n");
        return 1;
    }
    FaceTracker tracker;
    std::string error;
    if (!tracker.init(error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    printf("image %dx%d\n", img.width, img.height);
    // A second pass tracks from the first one's landmarks, like with video.
    for (int pass = 0; pass < 2; pass++) {
        FaceResult r = tracker.process(img);
        printf("pass %d: found %d, presence %.3f, roi (%.1f, %.1f) size %.1f rot %.1f deg, detect %.1f ms, landmarks %.1f ms\n",
            pass, r.found, r.presence, r.roi.cx, r.roi.cy, r.roi.size, r.roi.rotation * 57.2958f, r.detect_ms,
            r.landmark_ms);
        if (!r.found) {
            continue;
        }
        float yaw, pitch, roll;
        head_angles(r.rotation, Mat3(), yaw, pitch, roll);
        printf("  yaw %.1f, pitch %.1f, roll %.1f\n", yaw, pitch, roll);
        printf("  nose tip (%.1f, %.1f, %.1f), right eye (%.1f, %.1f), left eye (%.1f, %.1f)\n", r.landmarks[1].x,
            r.landmarks[1].y, r.landmarks[1].z, r.landmarks[33].x, r.landmarks[33].y, r.landmarks[263].x,
            r.landmarks[263].y);
        if (csv && pass == 1) {
            FILE* f = nullptr;
            if (_wfopen_s(&f, csv, L"w") != 0) {
                return 1;
            }
            for (const Vec3& p : r.landmarks) {
                fprintf(f, "%f,%f,%f\n", p.x, p.y, p.z);
            }
            fclose(f);
        }
    }
    return 0;
}

static Mat3 rot(int axis, float degrees) {
    const float a = degrees * 3.14159265f / 180, c = std::cos(a), s = std::sin(a);
    Mat3 r;
    const int i = (axis + 1) % 3, j = (axis + 2) % 3;
    r.m[i][i] = c;
    r.m[i][j] = -s;
    r.m[j][i] = s;
    r.m[j][j] = c;
    return r;
}

// Head rotation for angles in the person's terms (see head_angles()).
static Mat3 head(float yaw, float pitch, float roll) {
    return rot(1, -yaw) * rot(0, -pitch) * rot(2, roll);
}

static int cmd_pose() {
    FaceTracker tracker;
    std::string error;
    if (!tracker.init(error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    // The synthetic views are orthographic: no perspective to correct.
    tracker.set_fov(1e-3f);
    const int w = 1280, h = 720;
    const float cases[][3] = { { 0, 0, 0 }, { 20, 0, 0 }, { -30, 0, 0 }, { 0, 15, 0 }, { 0, -25, 0 }, { 0, 0, 20 },
        { 25, 15, -10 }, { -40, -20, 30 } };
    float worst = 0;
    for (const auto& c : cases) {
        // An orthographic view of the canonical face, in the middle of the image.
        const Mat3 r = head(c[0], c[1], c[2]);
        std::vector<Vec3> lm;
        for (const Vec3& p : tracker.canonical_face()) {
            const Vec3 q = r * p;
            lm.push_back({ w / 2.0f + q.x * 15, h / 2.0f - q.y * 15, -q.z * 15 });
        }
        float yaw, pitch, roll;
        head_angles(tracker.head_rotation(lm, w, h), Mat3(), yaw, pitch, roll);
        printf("expected %6.1f %6.1f %6.1f  got %6.1f %6.1f %6.1f\n", c[0], c[1], c[2], yaw, pitch, roll);
        worst = std::max({ worst, std::abs(yaw - c[0]), std::abs(pitch - c[1]), std::abs(roll - c[2]) });
        // Relative to a neutral that's turned and tilted.
        const Mat3 neutral = head(15, -10, 5);
        head_angles(neutral * r, neutral, yaw, pitch, roll);
        worst = std::max({ worst, std::abs(yaw - c[0]), std::abs(pitch - c[1]), std::abs(roll - c[2]) });
    }
    printf("worst error %.3f degrees\n", worst);
    return worst < 0.5f ? 0 : 1;
}

static int cmd_bench() {
    for (const char* which : { "detect", "landmark" }) {
        TfliteModel model;
        if (!load_model(which, model)) {
            return 1;
        }
        size_t n = 1;
        for (int d : model.input_shape()) {
            n *= d;
        }
        for (size_t i = 0; i < n; i++) {
            model.input()[i] = (float) ((i * 7919) % 255) / 255.0f;
        }
        model.run();
        const int runs = 50;
        const double t0 = now_ms();
        for (int i = 0; i < runs; i++) {
            model.run();
        }
        printf("%s: %.2f ms\n", which, (now_ms() - t0) / runs);
    }
    return 0;
}

int main(int argc, char** argv) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    int wargc;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "model" && argc == 5) {
        return cmd_model(argv[2], argv[3], argv[4]);
    }
    if (cmd == "image" && argc >= 3) {
        return cmd_image(wargv[2], argc >= 4 ? wargv[3] : nullptr);
    }
    if (cmd == "pose") {
        return cmd_pose();
    }
    if (cmd == "bench") {
        return cmd_bench();
    }
    fprintf(stderr, "usage: see the top of test_main.cpp\n");
    return 2;
}
