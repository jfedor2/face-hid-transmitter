// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tflite.h"

constexpr int LANDMARK_COUNT = 468;
// The region the landmark model looks at, relative to the face's size.
constexpr float FACE_ROI_SCALE = 1.5f;

// A camera frame: 32-bit BGRX pixels, top row first.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // width * height * 4
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

// 3x3 rotation, row major.
struct Mat3 {
    float m[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

    Vec3 operator*(const Vec3& v) const {
        return { m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z };
    }
    Mat3 operator*(const Mat3& o) const;
    Mat3 transposed() const;
};

// A square region of the image, rotated around its center.
struct Roi {
    float cx = 0, cy = 0;   // center, pixels
    float size = 0;         // side, pixels
    float rotation = 0;     // radians, clockwise on screen
};

struct FaceResult {
    bool found = false;
    // Image pixel coordinates; z is depth in the same units, larger = farther
    // from the camera.
    std::vector<Vec3> landmarks;
    float presence = 0.0f;  // the landmark model's confidence that it's a face
    Roi roi;                // where the landmarks were looked for
    // Head orientation in camera space (x right, y up, z towards the camera):
    // maps the face's own axes (x to the face's left, y up, z out of the face)
    // to camera space. Identity = looking straight into the camera.
    Mat3 rotation;
    bool detected = false;  // the face detector ran on this frame
    float detect_ms = 0.0f;
    float landmark_ms = 0.0f;
};

// The model files: face_detection_short_range.tflite, face_landmark.tflite
// and canonical_face_model.obj.
struct FaceModels {
    const uint8_t* detector = nullptr;
    size_t detector_size = 0;
    const uint8_t* landmarker = nullptr;
    size_t landmarker_size = 0;
    const char* canonical_obj = nullptr;
    size_t canonical_obj_size = 0;
};

// Finds a face with the MediaPipe face detector, then follows it from frame to
// frame with the face landmark model (468 points), which also says when it's
// lost. Not thread safe: use from one thread.
class FaceTracker {
public:
    // Prepares the models. The data is only used during the call.
    bool init(const FaceModels& models, std::string& error);
#ifdef _WIN32
    // Loads the models from the resources of the exe or DLL this code is
    // linked into (see resources.h).
    bool init(std::string& error);
#endif

    // Horizontal field of view of the camera, for correcting the perspective
    // of a face away from the image center.
    void set_fov(float degrees) { fov_degrees_ = degrees; }

    FaceResult process(const Image& image);
    void reset() { tracking_ = false; }

    // Head orientation (see FaceResult::rotation) from landmarks in an image
    // of the given size.
    Mat3 head_rotation(const std::vector<Vec3>& landmarks, int width, int height) const;
    // The canonical face model the landmarks are fitted to.
    const std::vector<Vec3>& canonical_face() const { return canonical_; }

private:
    struct Anchor {
        float x, y;
    };
    bool detect(const Image& image, Roi& roi);
    bool landmarks(const Image& image, const Roi& roi, FaceResult& result);

    TfliteModel detector_;
    TfliteModel landmarker_;
    std::vector<Anchor> anchors_;
    std::vector<Vec3> canonical_;  // canonical face model, same axes as the face in camera space
    std::vector<float> pose_weights_;
    bool tracking_ = false;
    Roi roi_;
    float fov_degrees_ = 60.0f;
};

// Yaw, pitch, roll in degrees of the head rotation relative to a neutral one,
// in the neutral head's frame. Yaw > 0: turned to the person's right. Pitch > 0:
// looking up. Roll > 0: tilted towards the right shoulder.
void head_angles(const Mat3& rotation, const Mat3& neutral, float& yaw, float& pitch, float& roll);

#ifdef _WIN32
// For tests: loads an image file (anything WIC can decode).
bool load_image_file(const std::wstring& path, Image& image);
#endif
