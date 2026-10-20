// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "face.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <locale>
#include <sstream>


static constexpr float PI = 3.14159265358979f;

// Face detection (MediaPipe face_detection_short_range)
static constexpr int DETECT_SIZE = 128;
static constexpr int DETECT_ANCHORS = 896;
static constexpr int DETECT_VALUES = 16;  // box (4) + 6 keypoints (x, y)
static constexpr float DETECT_THRESHOLD = 0.5f;
// Face landmarks (MediaPipe face_landmark)
static constexpr int LANDMARK_SIZE = 192;
static constexpr float PRESENCE_THRESHOLD = 0.5f;
// Eye corners, for the face's rotation in the image.
static constexpr int RIGHT_EYE_OUTER = 33;
static constexpr int LEFT_EYE_OUTER = 263;
// Only points above this height (canonical model units, the nose tip is at
// -1.1) are used for the head pose, so the mouth and jaw moving don't move it.
static constexpr float POSE_MAX_Y = -2.5f;

Mat3 Mat3::operator*(const Mat3& o) const {
    Mat3 r;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
        }
    }
    return r;
}

Mat3 Mat3::transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r.m[i][j] = m[j][i];
        }
    }
    return r;
}

static Vec3 operator-(const Vec3& a, const Vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

static Vec3 operator*(const Vec3& a, float s) {
    return { a.x * s, a.y * s, a.z * s };
}

static float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

static Vec3 normalized(const Vec3& v) {
    const float n = std::sqrt(dot(v, v));
    return n > 0 ? v * (1.0f / n) : v;
}

// Rotation by angle around a unit axis.
static Mat3 axis_angle(const Vec3& a, float angle) {
    const float c = std::cos(angle), s = std::sin(angle), t = 1 - c;
    Mat3 r;
    r.m[0][0] = t * a.x * a.x + c;
    r.m[0][1] = t * a.x * a.y - s * a.z;
    r.m[0][2] = t * a.x * a.z + s * a.y;
    r.m[1][0] = t * a.x * a.y + s * a.z;
    r.m[1][1] = t * a.y * a.y + c;
    r.m[1][2] = t * a.y * a.z - s * a.x;
    r.m[2][0] = t * a.x * a.z - s * a.y;
    r.m[2][1] = t * a.y * a.z + s * a.x;
    r.m[2][2] = t * a.z * a.z + c;
    return r;
}

static float normalize_radians(float a) {
    return a - 2 * PI * std::floor((a + PI) / (2 * PI));
}

static float sigmoid(float x) {
    x = std::clamp(x, -100.0f, 100.0f);
    return 1.0f / (1.0f + std::exp(-x));
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Fills a w x h RGB tensor from the image through an affine map: the center of
// tensor pixel (i, j) comes from image point origin + (i + 0.5) * ax + (j + 0.5) * ay
// (in pixels). Values are 0..1 scaled by scale and shifted by offset. Points
// outside the image are clamped to its edge, or black if !clamp_edges.
// Supersamples when shrinking so small faces in big frames don't alias.
static void sample_affine(const Image& img, float* out, int w, int h, float ox, float oy, float axx, float axy,
    float ayx, float ayy, bool clamp_edges, float scale, float offset) {
    const float step = std::max(std::hypot(axx, axy), std::hypot(ayx, ayy));
    const int n = std::clamp((int) std::ceil(step), 1, 4);
    const float inv = 1.0f / (n * n) / 255.0f;
    const int iw = img.width, ih = img.height;
    const uint8_t* px = img.pixels.data();
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            float acc[3] = { 0, 0, 0 };
            for (int sj = 0; sj < n; sj++) {
                for (int si = 0; si < n; si++) {
                    const float u = i + (si + 0.5f) / n;
                    const float v = j + (sj + 0.5f) / n;
                    float x = ox + u * axx + v * ayx - 0.5f;
                    float y = oy + u * axy + v * ayy - 0.5f;
                    if (!clamp_edges && (x < -0.5f || y < -0.5f || x > iw - 0.5f || y > ih - 0.5f)) {
                        continue;
                    }
                    x = std::clamp(x, 0.0f, (float) iw - 1);
                    y = std::clamp(y, 0.0f, (float) ih - 1);
                    const int x0 = std::min((int) x, iw - 2 < 0 ? 0 : iw - 2);
                    const int y0 = std::min((int) y, ih - 2 < 0 ? 0 : ih - 2);
                    const int x1 = std::min(x0 + 1, iw - 1);
                    const int y1 = std::min(y0 + 1, ih - 1);
                    const float fx = x - x0, fy = y - y0;
                    const uint8_t* p00 = px + ((size_t) y0 * iw + x0) * 4;
                    const uint8_t* p01 = px + ((size_t) y0 * iw + x1) * 4;
                    const uint8_t* p10 = px + ((size_t) y1 * iw + x0) * 4;
                    const uint8_t* p11 = px + ((size_t) y1 * iw + x1) * 4;
                    for (int c = 0; c < 3; c++) {
                        const float top = p00[c] + (p01[c] - p00[c]) * fx;
                        const float bottom = p10[c] + (p11[c] - p10[c]) * fx;
                        acc[2 - c] += top + (bottom - top) * fy;  // BGR -> RGB
                    }
                }
            }
            float* o = out + ((size_t) j * w + i) * 3;
            for (int c = 0; c < 3; c++) {
                o[c] = acc[c] * inv * scale + offset;
            }
        }
    }
}

bool FaceTracker::init(const FaceModels& models, std::string& error) {
    if (!models.detector || !detector_.load(models.detector, models.detector_size)) {
        error = "face detection model: " + detector_.error();
        return false;
    }
    if (!models.landmarker || !landmarker_.load(models.landmarker, models.landmarker_size)) {
        error = "face landmark model: " + landmarker_.error();
        return false;
    }
    if (detector_.input_shape() != std::vector<int>{ 1, DETECT_SIZE, DETECT_SIZE, 3 } ||
        detector_.output_count() != 2 || detector_.output_size(0) != DETECT_ANCHORS * DETECT_VALUES ||
        detector_.output_size(1) != DETECT_ANCHORS) {
        error = "face detection model has unexpected inputs or outputs";
        return false;
    }
    if (landmarker_.input_shape() != std::vector<int>{ 1, LANDMARK_SIZE, LANDMARK_SIZE, 3 } ||
        landmarker_.output_count() != 2 || landmarker_.output_size(0) != LANDMARK_COUNT * 3 ||
        landmarker_.output_size(1) != 1) {
        error = "face landmark model has unexpected inputs or outputs";
        return false;
    }

    if (!models.canonical_obj) {
        error = "canonical face model missing";
        return false;
    }
    canonical_.clear();
    std::istringstream obj(std::string((const char*) models.canonical_obj, models.canonical_obj_size));
    std::string line;
    while (std::getline(obj, line)) {
        if (line.rfind("v ", 0) == 0) {
            // Classic locale, so the decimal point is '.' whatever the host app set.
            std::istringstream coords(line.substr(2));
            coords.imbue(std::locale::classic());
            Vec3 v;
            if (coords >> v.x >> v.y >> v.z) {
                canonical_.push_back(v);
            }
        }
    }
    if (canonical_.size() != LANDMARK_COUNT) {
        error = "canonical face model has the wrong number of points";
        return false;
    }
    pose_weights_.resize(LANDMARK_COUNT);
    for (int i = 0; i < LANDMARK_COUNT; i++) {
        pose_weights_[i] = canonical_[i].y > POSE_MAX_Y ? 1.0f : 0.0f;
    }

    // SSD anchors for the short range model: 16x16 cells with 2 anchors at
    // stride 8, then 8x8 cells with 6 anchors at stride 16. The model has
    // fixed size anchors, so only the centers matter.
    anchors_.clear();
    const struct {
        int grid;
        int per_cell;
    } layers[] = { { 16, 2 }, { 8, 6 } };
    for (const auto& l : layers) {
        for (int y = 0; y < l.grid; y++) {
            for (int x = 0; x < l.grid; x++) {
                for (int k = 0; k < l.per_cell; k++) {
                    anchors_.push_back({ (x + 0.5f) / l.grid, (y + 0.5f) / l.grid });
                }
            }
        }
    }
    return true;
}

// Runs the face detector on the whole image (letterboxed to a square) and
// returns the region around the most confident face.
bool FaceTracker::detect(const Image& img, Roi& roi) {
    const float s = (float) DETECT_SIZE / std::max(img.width, img.height);  // tensor pixels per image pixel
    const float off_x = (DETECT_SIZE - img.width * s) / 2;
    const float off_y = (DETECT_SIZE - img.height * s) / 2;
    sample_affine(img, detector_.input(), DETECT_SIZE, DETECT_SIZE, -off_x / s, -off_y / s, 1 / s, 0, 0, 1 / s,
        false, 2.0f, -1.0f);
    detector_.run();
    const float* boxes = detector_.output(0);
    const float* scores = detector_.output(1);
    int best = -1;
    float best_score = DETECT_THRESHOLD;
    for (int i = 0; i < DETECT_ANCHORS; i++) {
        const float score = sigmoid(scores[i]);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    if (best < 0) {
        return false;
    }
    const float* b = boxes + best * DETECT_VALUES;
    const Anchor& a = anchors_[best];
    // Tensor pixels -> image pixels.
    auto to_x = [&](float v) { return (v - off_x) / s; };
    auto to_y = [&](float v) { return (v - off_y) / s; };
    const float cx = to_x(b[0] + a.x * DETECT_SIZE);
    const float cy = to_y(b[1] + a.y * DETECT_SIZE);
    const float w = b[2] / s;
    const float h = b[3] / s;
    // Keypoints 0 and 1 are the right and left eye.
    const float rx = to_x(b[4] + a.x * DETECT_SIZE), ry = to_y(b[5] + a.y * DETECT_SIZE);
    const float lx = to_x(b[6] + a.x * DETECT_SIZE), ly = to_y(b[7] + a.y * DETECT_SIZE);
    roi.cx = cx;
    roi.cy = cy;
    roi.size = std::max(w, h) * FACE_ROI_SCALE;
    roi.rotation = normalize_radians(-std::atan2(-(ly - ry), lx - rx));
    return roi.size > 4;
}

bool FaceTracker::landmarks(const Image& img, const Roi& roi, FaceResult& result) {
    const float c = std::cos(roi.rotation), s = std::sin(roi.rotation);
    const float k = roi.size / LANDMARK_SIZE;
    // Tensor x axis runs along (c, s), y axis along (-s, c).
    const float ox = roi.cx - roi.size / 2 * c + roi.size / 2 * s;
    const float oy = roi.cy - roi.size / 2 * s - roi.size / 2 * c;
    sample_affine(img, landmarker_.input(), LANDMARK_SIZE, LANDMARK_SIZE, ox, oy, k * c, k * s, -k * s, k * c, true,
        1.0f, 0.0f);
    landmarker_.run();
    result.presence = sigmoid(landmarker_.output(1)[0]);
    result.roi = roi;
    if (result.presence < PRESENCE_THRESHOLD) {
        return false;
    }
    const float* p = landmarker_.output(0);
    result.landmarks.resize(LANDMARK_COUNT);
    for (int i = 0; i < LANDMARK_COUNT; i++) {
        const float u = p[i * 3], v = p[i * 3 + 1];
        result.landmarks[i] = { ox + u * k * c - v * k * s, oy + u * k * s + v * k * c, p[i * 3 + 2] * k };
    }
    return true;
}

// Largest eigenvector of a symmetric 4x4 matrix (Jacobi rotations).
static void largest_eigenvector(double a[4][4], double out[4]) {
    double v[4][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
    for (int sweep = 0; sweep < 30; sweep++) {
        double off = 0;
        for (int p = 0; p < 4; p++) {
            for (int q = p + 1; q < 4; q++) {
                off += a[p][q] * a[p][q];
            }
        }
        if (off < 1e-20) {
            break;
        }
        for (int p = 0; p < 4; p++) {
            for (int q = p + 1; q < 4; q++) {
                if (std::abs(a[p][q]) < 1e-30) {
                    continue;
                }
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 4; k++) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; k++) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 4; k++) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (a[i][i] > a[best][best]) {
            best = i;
        }
    }
    for (int k = 0; k < 4; k++) {
        out[k] = v[k][best];
    }
}

// Rotation that best maps points p onto points q (weighted, any scale), by
// Horn's quaternion method. Also returns the scale.
static Mat3 fit_rotation(const std::vector<Vec3>& p, const std::vector<Vec3>& q, const std::vector<float>& w,
    float& scale) {
    double sw = 0;
    double pc[3] = {}, qc[3] = {};
    for (size_t i = 0; i < p.size(); i++) {
        sw += w[i];
        pc[0] += w[i] * p[i].x, pc[1] += w[i] * p[i].y, pc[2] += w[i] * p[i].z;
        qc[0] += w[i] * q[i].x, qc[1] += w[i] * q[i].y, qc[2] += w[i] * q[i].z;
    }
    for (int k = 0; k < 3; k++) {
        pc[k] /= sw;
        qc[k] /= sw;
    }
    double S[3][3] = {};
    double pp = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const double a[3] = { p[i].x - pc[0], p[i].y - pc[1], p[i].z - pc[2] };
        const double b[3] = { q[i].x - qc[0], q[i].y - qc[1], q[i].z - qc[2] };
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                S[r][c] += w[i] * a[r] * b[c];
            }
        }
        pp += w[i] * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }
    const double sxx = S[0][0], sxy = S[0][1], sxz = S[0][2];
    const double syx = S[1][0], syy = S[1][1], syz = S[1][2];
    const double szx = S[2][0], szy = S[2][1], szz = S[2][2];
    double n[4][4] = {
        { sxx + syy + szz, syz - szy, szx - sxz, sxy - syx },
        { syz - szy, sxx - syy - szz, sxy + syx, szx + sxz },
        { szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy },
        { sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz },
    };
    double qt[4];
    largest_eigenvector(n, qt);
    const double qw = qt[0], qx = qt[1], qy = qt[2], qz = qt[3];
    Mat3 r;
    r.m[0][0] = (float) (qw * qw + qx * qx - qy * qy - qz * qz);
    r.m[0][1] = (float) (2 * (qx * qy - qw * qz));
    r.m[0][2] = (float) (2 * (qx * qz + qw * qy));
    r.m[1][0] = (float) (2 * (qx * qy + qw * qz));
    r.m[1][1] = (float) (qw * qw - qx * qx + qy * qy - qz * qz);
    r.m[1][2] = (float) (2 * (qy * qz - qw * qx));
    r.m[2][0] = (float) (2 * (qx * qz - qw * qy));
    r.m[2][1] = (float) (2 * (qy * qz + qw * qx));
    r.m[2][2] = (float) (qw * qw - qx * qx - qy * qy + qz * qz);
    // scale = sum(q' . R p') / sum(|p'|^2)
    double qrp = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const Vec3 a = r * Vec3{ (float) (p[i].x - pc[0]), (float) (p[i].y - pc[1]), (float) (p[i].z - pc[2]) };
        qrp += w[i] * (a.x * (q[i].x - qc[0]) + a.y * (q[i].y - qc[1]) + a.z * (q[i].z - qc[2]));
    }
    scale = pp > 0 ? (float) (qrp / pp) : 1.0f;
    return r;
}

Mat3 FaceTracker::head_rotation(const std::vector<Vec3>& landmarks, int width, int height) const {
    // Landmarks in camera axes (y up, z towards the camera).
    std::vector<Vec3> q(LANDMARK_COUNT);
    Vec3 center;
    float sw = 0;
    for (int i = 0; i < LANDMARK_COUNT; i++) {
        const Vec3& l = landmarks[i];
        q[i] = { l.x, -l.y, -l.z };
        if (pose_weights_[i] > 0) {
            center.x += l.x;
            center.y += l.y;
            sw += 1;
        }
    }
    center.x /= sw;
    center.y /= sw;

    // Iteratively reweighted, so points that moved relative to the rest
    // (eyebrows, blinks) count less.
    std::vector<float> w = pose_weights_;
    Mat3 r;
    for (int iter = 0; iter < 3; iter++) {
        float scale;
        r = fit_rotation(canonical_, q, w, scale);
        if (iter == 2) {
            break;
        }
        // residuals, relative to the fitted model
        Vec3 pc, qc;
        float total = 0;
        for (int i = 0; i < LANDMARK_COUNT; i++) {
            pc.x += w[i] * canonical_[i].x, pc.y += w[i] * canonical_[i].y, pc.z += w[i] * canonical_[i].z;
            qc.x += w[i] * q[i].x, qc.y += w[i] * q[i].y, qc.z += w[i] * q[i].z;
            total += w[i];
        }
        pc = pc * (1 / total);
        qc = qc * (1 / total);
        std::vector<float> res(LANDMARK_COUNT, 0.0f);
        std::vector<float> used;
        for (int i = 0; i < LANDMARK_COUNT; i++) {
            if (pose_weights_[i] == 0) {
                continue;
            }
            const Vec3 d = (r * (canonical_[i] - pc)) * scale - (q[i] - qc);
            res[i] = std::sqrt(dot(d, d));
            used.push_back(res[i]);
        }
        std::nth_element(used.begin(), used.begin() + used.size() / 2, used.end());
        const float sigma = std::max(used[used.size() / 2] * 1.5f, 1e-3f);
        for (int i = 0; i < LANDMARK_COUNT; i++) {
            if (pose_weights_[i] > 0) {
                const float e = res[i] / sigma;
                w[i] = pose_weights_[i] / (1 + e * e);
            }
        }
    }

    // The landmark model sees the face as if it were in the middle of the
    // view. A face off to the side is seen at an angle, so a face looking
    // straight into the lens is really looking back along the ray to the camera.
    const float f = width / 2.0f / std::tan(fov_degrees_ * PI / 360);
    const Vec3 to_camera = normalized({ -(center.x - width / 2.0f) / f, (center.y - height / 2.0f) / f, 1 });
    const Vec3 axis = cross({ 0, 0, 1 }, to_camera);
    const float axis_len = std::sqrt(dot(axis, axis));
    if (axis_len > 1e-6f) {
        r = axis_angle(axis * (1 / axis_len), std::asin(std::min(axis_len, 1.0f))) * r;
    }
    return r;
}

FaceResult FaceTracker::process(const Image& img) {
    FaceResult result;
    if (img.width < 2 || img.height < 2) {
        return result;
    }
    // Follow the face from the last frame's landmarks; if it's lost, look for
    // it with the detector (in the same frame).
    for (int attempt = 0; attempt < 2; attempt++) {
        Roi roi = roi_;
        if (!tracking_) {
            const double t0 = now_ms();
            const bool found = detect(img, roi);
            result.detect_ms = (float) (now_ms() - t0);
            result.detected = true;
            if (!found) {
                return result;
            }
        }
        const double t0 = now_ms();
        const bool found = landmarks(img, roi, result);
        result.landmark_ms += (float) (now_ms() - t0);
        if (found) {
            break;
        }
        if (!tracking_) {
            return result;
        }
        tracking_ = false;
    }

    result.found = true;
    result.rotation = head_rotation(result.landmarks, img.width, img.height);

    // Region for the next frame: around the landmarks, upright.
    const Vec3& r = result.landmarks[RIGHT_EYE_OUTER];
    const Vec3& l = result.landmarks[LEFT_EYE_OUTER];
    float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
    for (const Vec3& p : result.landmarks) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    roi_.cx = (min_x + max_x) / 2;
    roi_.cy = (min_y + max_y) / 2;
    roi_.size = std::max(max_x - min_x, max_y - min_y) * FACE_ROI_SCALE;
    roi_.rotation = normalize_radians(-std::atan2(-(l.y - r.y), l.x - r.x));
    tracking_ = roi_.size > 4;
    return result;
}

void head_angles(const Mat3& rotation, const Mat3& neutral, float& yaw, float& pitch, float& roll) {
    // The face's axes in the neutral face's frame: x to the face's left, y up,
    // z out of the face.
    const Mat3 rel = neutral.transposed() * rotation;
    const Vec3 forward = rel * Vec3{ 0, 0, 1 };
    const Vec3 up = rel * Vec3{ 0, 1, 0 };
    yaw = std::atan2(-forward.x, forward.z);
    pitch = std::atan2(forward.y, std::hypot(forward.x, forward.z));
    // Roll: how far up is turned around forward from where it would be with
    // the head not tilted.
    Vec3 level = Vec3{ 0, 1, 0 } - forward * forward.y;
    if (dot(level, level) < 1e-8f) {
        roll = 0;
    } else {
        level = normalized(level);
        roll = std::atan2(dot(cross(level, up), forward), dot(level, up));
    }
    yaw *= 180 / PI;
    pitch *= 180 / PI;
    roll *= 180 / PI;
}
