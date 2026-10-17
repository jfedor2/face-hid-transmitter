// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include <winsock2.h>
#include <windows.h>

#include <d3d11.h>
#include <mfapi.h>
#include <objbase.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include "camera.h"
#include "config.h"
#include "face.h"
#include "mapping.h"
#include "sender.h"
#include "tracker.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static ID3D11Device* g_device = nullptr;
static ID3D11DeviceContext* g_device_context = nullptr;
static IDXGISwapChain* g_swap_chain = nullptr;
static ID3D11RenderTargetView* g_render_target_view = nullptr;
static bool g_swap_chain_occluded = false;
static UINT g_resize_width = 0;
static UINT g_resize_height = 0;
static float g_pending_dpi_scale = 0.0f;

static const ImVec4 COLOR_ERROR(1.0f, 0.42f, 0.42f, 1.0f);
static const ImVec4 COLOR_OK(0.45f, 0.9f, 0.55f, 1.0f);
static const ImVec4 COLOR_WARN(1.0f, 0.8f, 0.35f, 1.0f);

// Overlay colors.
static const ImU32 COLOR_MESH_POINT = IM_COL32(255, 255, 255, 110);
static const ImU32 COLOR_MESH_POINT_PRIVATE = IM_COL32(120, 200, 255, 150);
static const ImU32 COLOR_CONTOUR = IM_COL32(80, 220, 255, 230);
static const ImU32 COLOR_DIRECTION = IM_COL32(255, 170, 60, 255);

// Face contours (MediaPipe face mesh landmark indices).
static const int CONTOUR_FACE_OVAL[] = { 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10 };
static const int CONTOUR_LIPS_OUTER[] = { 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185, 61 };
static const int CONTOUR_LIPS_INNER[] = { 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
    82, 81, 80, 191, 78 };
static const int CONTOUR_RIGHT_EYE[] = { 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    33 };
static const int CONTOUR_LEFT_EYE[] = { 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388,
    466, 263 };
static const int CONTOUR_RIGHT_BROW[] = { 70, 63, 105, 66, 107 };
static const int CONTOUR_LEFT_BROW[] = { 300, 293, 334, 296, 336 };
static const int CONTOUR_NOSE[] = { 168, 6, 197, 195, 5, 4, 1 };
static const struct {
    const int* points;
    int count;
} CONTOURS[] = {
    { CONTOUR_FACE_OVAL, IM_ARRAYSIZE(CONTOUR_FACE_OVAL) },
    { CONTOUR_LIPS_OUTER, IM_ARRAYSIZE(CONTOUR_LIPS_OUTER) },
    { CONTOUR_LIPS_INNER, IM_ARRAYSIZE(CONTOUR_LIPS_INNER) },
    { CONTOUR_RIGHT_EYE, IM_ARRAYSIZE(CONTOUR_RIGHT_EYE) },
    { CONTOUR_LEFT_EYE, IM_ARRAYSIZE(CONTOUR_LEFT_EYE) },
    { CONTOUR_RIGHT_BROW, IM_ARRAYSIZE(CONTOUR_RIGHT_BROW) },
    { CONTOUR_LEFT_BROW, IM_ARRAYSIZE(CONTOUR_LEFT_BROW) },
    { CONTOUR_NOSE, IM_ARRAYSIZE(CONTOUR_NOSE) },
};
static constexpr int NOSE_TIP = 1;

// The part of the camera image shown in the preview, in image pixels.
struct View {
    float cx = 0, cy = 0;
    float h = 0;  // width is h * aspect
};

// Application state, owned by the UI thread.
struct App {
    Config config;
    std::wstring config_path;
    bool running = false;
    std::vector<CameraDevice> cameras;
    std::string tracker_error;

    bool config_dirty = false;
    ULONGLONG config_dirty_since = 0;
    HWND main_hwnd = nullptr;

    Sender sender;
    HeadTracker tracker;
    Camera camera;

    // Preview
    std::shared_ptr<const TrackerSnapshot> snapshot;
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* texture_view = nullptr;
    int texture_w = 0, texture_h = 0;
    uint64_t uploaded_seq = 0;
    View view;
    bool view_valid = false;
    double face_seen_time = -1e9;

    void mark_dirty() {
        if (!config_dirty) {
            config_dirty_since = GetTickCount64();
        }
        config_dirty = true;
    }

    // Call after any change to the mapping or neutral.
    void mapping_changed() {
        TrackerSettings s;
        for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
            s.axes[a] = config.axes[a];
        }
        s.neutral = config.neutral_set ? config.neutral : Mat3();
        s.smoothing = config.smoothing;
        tracker.set_settings(s);
        mark_dirty();
    }
};

static App g_app;

// ---------------------------------------------------------------------------
// D3D11 boilerplate (from the Dear ImGui example_win32_directx11)

static void create_render_target() {
    ID3D11Texture2D* back_buffer = nullptr;
    g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    if (back_buffer != nullptr) {
        g_device->CreateRenderTargetView(back_buffer, nullptr, &g_render_target_view);
        back_buffer->Release();
    }
}

static void cleanup_render_target() {
    if (g_render_target_view != nullptr) {
        g_render_target_view->Release();
        g_render_target_view = nullptr;
    }
}

static bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL level;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2,
        D3D11_SDK_VERSION, &sd, &g_swap_chain, &g_device, &level, &g_device_context);
    if (hr == DXGI_ERROR_UNSUPPORTED) {
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2,
            D3D11_SDK_VERSION, &sd, &g_swap_chain, &g_device, &level, &g_device_context);
    }
    if (FAILED(hr)) {
        return false;
    }
    create_render_target();
    return true;
}

static void cleanup_device() {
    cleanup_render_target();
    if (g_swap_chain) {
        g_swap_chain->Release();
        g_swap_chain = nullptr;
    }
    if (g_device_context) {
        g_device_context->Release();
        g_device_context = nullptr;
    }
    if (g_device) {
        g_device->Release();
        g_device = nullptr;
    }
}

// ---------------------------------------------------------------------------

static void apply_style(float scale) {
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    ImGui::StyleColorsDark();
    style.Colors[ImGuiCol_PopupBg].w = 1.0f;
    style.WindowRounding = 0.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(8, 6);
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
}

static bool g_track_placement = false;  // set once the saved placement has been applied

static void capture_window_placement() {
    if (!g_track_placement) {
        return;
    }
    WINDOWPLACEMENT wp = { sizeof(wp) };
    if (!GetWindowPlacement(g_app.main_hwnd, &wp)) {
        return;
    }
    WindowRect w;
    w.valid = true;
    w.x = wp.rcNormalPosition.left;
    w.y = wp.rcNormalPosition.top;
    w.w = wp.rcNormalPosition.right - wp.rcNormalPosition.left;
    w.h = wp.rcNormalPosition.bottom - wp.rcNormalPosition.top;
    w.maximized = wp.showCmd == SW_SHOWMAXIMIZED ||
                  (wp.showCmd == SW_SHOWMINIMIZED && (wp.flags & WPF_RESTORETOMAXIMIZED));
    if (!(w == g_app.config.window)) {
        g_app.config.window = w;
        g_app.mark_dirty();
    }
}

static void save_now() {
    save_config(g_app.config_path, g_app.config);
    g_app.config_dirty = false;
}

static void push_button_colors(ImVec4 base) {
    ImGui::PushStyleColor(ImGuiCol_Button, base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(base.x * 1.2f, base.y * 1.2f, base.z * 1.2f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(base.x * 1.4f, base.y * 1.4f, base.z * 1.4f, 1));
}

static double now_seconds() {
    return ImGui::GetTime();
}

// ---------------------------------------------------------------------------
// Preview

static void release_texture() {
    App& app = g_app;
    if (app.texture_view) {
        app.texture_view->Release();
        app.texture_view = nullptr;
    }
    if (app.texture) {
        app.texture->Release();
        app.texture = nullptr;
    }
    app.texture_w = app.texture_h = 0;
}

// Copies the latest frame into the preview texture.
static void upload_frame(const Image& img) {
    App& app = g_app;
    if (img.width != app.texture_w || img.height != app.texture_h) {
        release_texture();
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = img.width;
        desc.Height = img.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(g_device->CreateTexture2D(&desc, nullptr, &app.texture)) ||
            FAILED(g_device->CreateShaderResourceView(app.texture, nullptr, &app.texture_view))) {
            release_texture();
            return;
        }
        app.texture_w = img.width;
        app.texture_h = img.height;
    }
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(g_device_context->Map(app.texture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        return;
    }
    const size_t row = (size_t) img.width * 4;
    for (int y = 0; y < img.height; y++) {
        uint8_t* dst = (uint8_t*) mapped.pData + (size_t) y * mapped.RowPitch;
        memcpy(dst, &img.pixels[y * row], row);
        // The fourth byte is padding, not alpha.
        for (int x = 0; x < img.width; x++) {
            dst[x * 4 + 3] = 255;
        }
    }
    g_device_context->Unmap(app.texture, 0);
}

// Moves the preview view towards the face (or the whole image).
static void update_view(const TrackerSnapshot* snap, int img_w, int img_h, float aspect) {
    App& app = g_app;
    const float full_h = std::min((float) img_h, img_w / aspect);
    View target{ img_w / 2.0f, img_h / 2.0f, full_h };
    const double now = now_seconds();
    if (snap && snap->face.found) {
        app.face_seen_time = now;
    }
    if (app.config.zoom_to_face && snap && !snap->face.landmarks.empty() && now - app.face_seen_time < 1.5) {
        float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
        for (const Vec3& p : snap->face.landmarks) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        target.h = std::min(std::max(max_y - min_y, (max_x - min_x) / aspect) * 1.4f, full_h);
        target.cx = (min_x + max_x) / 2;
        target.cy = (min_y + max_y) / 2;
    }
    // Keep it inside the image.
    const float half_w = target.h * aspect / 2, half_h = target.h / 2;
    target.cx = std::clamp(target.cx, half_w, std::max(half_w, img_w - half_w));
    target.cy = std::clamp(target.cy, half_h, std::max(half_h, img_h - half_h));

    if (!app.view_valid) {
        app.view = target;
        app.view_valid = true;
        return;
    }
    const float k = 1.0f - std::exp(-ImGui::GetIO().DeltaTime * 4.0f);
    app.view.cx += (target.cx - app.view.cx) * k;
    app.view.cy += (target.cy - app.view.cy) * k;
    app.view.h += (target.h - app.view.h) * k;
}

static void draw_preview(float height_limit) {
    App& app = g_app;
    Config& cfg = app.config;
    const float em = ImGui::GetFontSize();
    const TrackerSnapshot* snap = app.snapshot.get();
    const Image* img = snap && snap->frame ? snap->frame.get() : nullptr;

    const float aspect = img ? (float) img->width / img->height : 4.0f / 3.0f;
    float w = ImGui::GetContentRegionAvail().x;
    float h = w / aspect;
    if (h > height_limit) {
        h = std::max(height_limit, em * 6);
        w = h * aspect;
    }
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + w, p0.y + h);
    ImGui::Dummy(ImVec2(w, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, IM_COL32(16, 17, 20, 255), 4.0f);

    auto centered_text = [&](const char* text, ImU32 color) {
        const ImVec2 size = ImGui::CalcTextSize(text, nullptr, false, w - em * 2);
        dl->AddText(nullptr, 0.0f, ImVec2(p0.x + (w - size.x) / 2, p0.y + (h - size.y) / 2), color, text, nullptr,
            w - em * 2);
    };

    if (!img) {
        const std::string err = app.camera.error();
        if (!app.tracker_error.empty()) {
            centered_text(("Face tracking failed to start: " + app.tracker_error).c_str(), ImGui::GetColorU32(COLOR_ERROR));
        } else if (!err.empty()) {
            centered_text(err.c_str(), ImGui::GetColorU32(COLOR_ERROR));
        } else {
            centered_text("Starting the camera...", IM_COL32(160, 160, 170, 255));
        }
        return;
    }

    update_view(snap, img->width, img->height, aspect);
    const View& v = app.view;
    const float view_w = v.h * aspect;
    const float x0 = v.cx - view_w / 2, y0 = v.cy - v.h / 2;
    const float sx = w / view_w, sy = h / v.h;
    auto to_screen = [&](float x, float y) {
        const float px = (x - x0) * sx;
        return ImVec2(cfg.mirror ? p1.x - px : p0.x + px, p0.y + (y - y0) * sy);
    };

    dl->PushClipRect(p0, p1, true);
    if (!cfg.privacy && app.texture_view) {
        float u0 = x0 / img->width, u1 = (x0 + view_w) / img->width;
        if (cfg.mirror) {
            std::swap(u0, u1);
        }
        dl->AddImage((ImTextureID) (intptr_t) app.texture_view, p0, p1, ImVec2(u0, y0 / img->height),
            ImVec2(u1, (y0 + v.h) / img->height));
    }

    const FaceResult& face = snap->face;
    if (!face.landmarks.empty()) {
        // Scale the overlay with how big the face is on screen.
        const float face_px = face.roi.size / FACE_ROI_SCALE * sx;
        const float dot = std::clamp(face_px / 260.0f, 0.8f, 2.2f);
        const float thick = std::clamp(face_px / 160.0f, 1.0f, 3.0f);
        const ImU32 point_color = cfg.privacy ? COLOR_MESH_POINT_PRIVATE : COLOR_MESH_POINT;
        for (const Vec3& p : face.landmarks) {
            dl->AddCircleFilled(to_screen(p.x, p.y), dot, point_color, 6);
        }
        for (const auto& c : CONTOURS) {
            ImVec2 pts[64];
            for (int i = 0; i < c.count; i++) {
                const Vec3& p = face.landmarks[c.points[i]];
                pts[i] = to_screen(p.x, p.y);
            }
            dl->AddPolyline(pts, c.count, COLOR_CONTOUR, ImDrawFlags_None, thick);
        }
        // Where the face points (in the image plane).
        const Vec3 forward = face.rotation * Vec3{ 0, 0, 1 };
        const Vec3& nose = face.landmarks[NOSE_TIP];
        const float len = face.roi.size / FACE_ROI_SCALE * 0.6f;
        const ImVec2 a = to_screen(nose.x, nose.y);
        const ImVec2 b = to_screen(nose.x + forward.x * len, nose.y - forward.y * len);
        dl->AddLine(a, b, COLOR_DIRECTION, thick * 1.5f);
        dl->AddCircleFilled(b, thick * 2.5f, COLOR_DIRECTION);
    }
    dl->PopClipRect();

    // Status in the corners.
    const ImVec2 pad(em * 0.5f, em * 0.3f);
    if (!face.found) {
        const char* text = "No face found";
        const ImVec2 size = ImGui::CalcTextSize(text);
        const ImVec2 tp(p0.x + (w - size.x) / 2, p0.y + pad.y * 2);
        dl->AddRectFilled(ImVec2(tp.x - pad.x, tp.y - pad.y), ImVec2(tp.x + size.x + pad.x, tp.y + size.y + pad.y),
            IM_COL32(0, 0, 0, 170), 4.0f);
        dl->AddText(tp, ImGui::GetColorU32(COLOR_WARN), text);
    }
    if (cfg.privacy) {
        const char* text = "Privacy: camera image hidden";
        dl->AddText(ImVec2(p0.x + pad.x, p1.y - em - pad.y), IM_COL32(150, 150, 160, 255), text);
    }
}

// ---------------------------------------------------------------------------
// Output

static void draw_stick_indicator(Stick stick, const float (&value)[2]) {
    const float em = ImGui::GetFontSize();
    const float radius = em * 1.3f;
    const ImVec2 size(radius * 2, radius * 2);
    ImGui::BeginGroup();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 c(p.x + radius, p.y + radius);
    const Color3& col = STICK_COLORS[(int) stick];
    const bool centered = stick_byte(value[0]) == 128 && stick_byte(value[1]) == 128;
    dl->AddCircleFilled(c, radius, IM_COL32(40, 42, 48, 255));
    dl->AddCircle(c, radius, ImGui::GetColorU32(ImVec4(col.r, col.g, col.b, 1)), 0, 1.5f);
    dl->AddLine(ImVec2(c.x - radius, c.y), ImVec2(c.x + radius, c.y), IM_COL32(255, 255, 255, 30));
    dl->AddLine(ImVec2(c.x, c.y - radius), ImVec2(c.x, c.y + radius), IM_COL32(255, 255, 255, 30));
    const ImVec2 dot(c.x + value[0] * (radius - 3), c.y + value[1] * (radius - 3));
    dl->AddCircleFilled(dot, em * 0.22f,
        centered ? IM_COL32(120, 120, 130, 255)
                 : ImGui::GetColorU32(ImVec4(col.r + (1 - col.r) * 0.4f, col.g + (1 - col.g) * 0.4f, col.b + (1 - col.b) * 0.4f, 1)));
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::TextUnformatted(STICK_NAMES[(int) stick]);
    ImGui::TextDisabled("%+.2f, %+.2f", value[0], -value[1]);
    ImGui::EndGroup();
    ImGui::EndGroup();
}

static void draw_output() {
    const GamepadState state = g_app.snapshot ? g_app.snapshot->state : GamepadState();
    const ImGuiStyle& style = ImGui::GetStyle();
    const float em = ImGui::GetFontSize();
    const float slot_w = em * 2.6f + style.ItemSpacing.x +
                         std::max(ImGui::CalcTextSize(STICK_NAMES[(int) Stick::Right]).x,
                             ImGui::CalcTextSize("-0.00, -0.00").x * 1.15f);
    const float x0 = ImGui::GetCursorPosX();
    draw_stick_indicator(Stick::Left, state.sticks[(int) Stick::Left]);
    ImGui::SameLine(x0 + slot_w + em);
    draw_stick_indicator(Stick::Right, state.sticks[(int) Stick::Right]);
    for (int t = 0; t < TRIGGER_COUNT; t++) {
        const Color3& c = BUTTON_COLORS[(int) trigger_button((Trigger) t)];
        char overlay[16];
        snprintf(overlay, sizeof(overlay), "%.2f", state.triggers[t]);
        if (t > 0) {
            ImGui::SameLine(0, em);
        }
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(TRIGGER_NAMES[t]);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(c.r, c.g, c.b, 1));
        ImGui::ProgressBar(state.triggers[t], ImVec2(em * 6, em * 1.1f), overlay);
        ImGui::PopStyleColor();
    }
    if (!g_app.running) {
        ImGui::TextDisabled("Stopped: the receiver gets nothing pressed.");
    }
}

// ---------------------------------------------------------------------------
// Mapping

// A head angle on a scale, with the deadzone shaded, the full output angles
// marked and the output filled in from the center.
static void draw_angle_bar(const AxisMapping& m, float angle, bool valid, float width) {
    const float em = ImGui::GetFontSize();
    const float h = em * 1.25f;
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + width, p0.y + h);
    ImGui::Dummy(ImVec2(width, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float scale = std::max(45.0f, m.range + 10.0f);  // degrees at each end
    const float cx = (p0.x + p1.x) / 2;
    auto x_of = [&](float deg) { return cx + std::clamp(deg / scale, -1.0f, 1.0f) * width / 2; };
    const bool used = m.target != AxisTarget::None;

    dl->AddRectFilled(p0, p1, IM_COL32(32, 34, 40, 255), 3.0f);
    if (used) {
        // Output fill
        const float out = valid ? axis_output(m, angle) : 0.0f;
        if (out != 0.0f) {
            const float edge = m.deadzone + std::abs(out) * (m.range - m.deadzone);
            const float sign = (angle < 0) ? -1.0f : 1.0f;
            const float a = x_of(sign * m.deadzone), b = x_of(sign * edge);
            dl->AddRectFilled(ImVec2(std::min(a, b), p0.y + 2), ImVec2(std::max(a, b), p1.y - 2),
                IM_COL32(60, 140, 220, 200), 2.0f);
        }
        // Deadzone
        dl->AddRectFilled(ImVec2(x_of(-m.deadzone), p0.y), ImVec2(x_of(m.deadzone), p1.y), IM_COL32(255, 255, 255, 22));
        // Full output
        for (float s : { -1.0f, 1.0f }) {
            const float x = x_of(s * m.range);
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), IM_COL32(255, 255, 255, 120), 1.5f);
        }
    }
    dl->AddLine(ImVec2(cx, p0.y), ImVec2(cx, p1.y), IM_COL32(255, 255, 255, 50));
    if (valid) {
        const float x = x_of(angle);
        dl->AddTriangleFilled(ImVec2(x - em * 0.35f, p0.y), ImVec2(x + em * 0.35f, p0.y), ImVec2(x, p0.y + em * 0.45f),
            IM_COL32(255, 200, 80, 255));
        dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), IM_COL32(255, 200, 80, 255), 2.0f);
    }
}

static void draw_axis(int a) {
    App& app = g_app;
    AxisMapping& m = app.config.axes[a];
    const float em = ImGui::GetFontSize();
    const TrackerSnapshot* snap = app.snapshot.get();
    const bool valid = snap && snap->face.found;
    const float angle = valid ? snap->angles[a] : 0.0f;
    bool changed = false;

    ImGui::PushID(a);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(HEAD_AXIS_NAMES[a]);
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", HEAD_AXIS_DESCRIPTIONS[a]);
    ImGui::SameLine();
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    char value[32];
    if (valid) {
        snprintf(value, sizeof(value), "%+.1f°", angle);
    } else {
        snprintf(value, sizeof(value), "--");
    }
    ImGui::SetCursorPosX(right - ImGui::CalcTextSize("+00.0°").x);
    ImGui::TextUnformatted(value);

    draw_angle_bar(m, angle, valid, ImGui::GetContentRegionAvail().x);

    const float label_w = em * 5.5f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Output");
    ImGui::SameLine(label_w);
    ImGui::SetNextItemWidth(em * 9);
    if (ImGui::BeginCombo("##target", AXIS_TARGET_NAMES[(int) m.target])) {
        for (int t = 0; t < AXIS_TARGET_COUNT; t++) {
            if (ImGui::Selectable(AXIS_TARGET_NAMES[t], (int) m.target == t)) {
                m.target = (AxisTarget) t;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip(
        "Sticks: turning right, looking up and tilting right move the stick right / up.\n"
        "L2 / R2: above neutral drives R2, below drives L2.\n"
        "L2 or R2 alone: driven by the right / up side only (invert for the other side).\n"
        "Several angles can drive the same output; they add up.");
    ImGui::SameLine(0, em);
    ImGui::BeginDisabled(m.target == AxisTarget::None);
    changed |= ImGui::Checkbox("Invert", &m.invert);
    ImGui::SameLine(0, em);
    const float out = valid && m.target != AxisTarget::None ? axis_output(m, angle) : 0.0f;
    ImGui::TextDisabled("output %+.2f", out);

    const float slider_w = ImGui::GetContentRegionAvail().x - label_w;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Sensitivity");
    ImGui::SameLine(label_w);
    ImGui::SetNextItemWidth(slider_w);
    if (ImGui::SliderFloat("##range", &m.range, 2.0f, 60.0f, "full at %.0f°")) {
        m.range = std::max(m.range, m.deadzone + 1.0f);
        changed = true;
    }
    ImGui::SetItemTooltip("How far from neutral the head must turn for full output.\nLower = more sensitive.");
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Deadzone");
    ImGui::SameLine(label_w);
    ImGui::SetNextItemWidth(slider_w);
    if (ImGui::SliderFloat("##deadzone", &m.deadzone, 0.0f, 15.0f, "%.1f°")) {
        m.range = std::max(m.range, m.deadzone + 1.0f);
        changed = true;
    }
    ImGui::SetItemTooltip("How far from neutral the head can move with no output.");
    ImGui::EndDisabled();
    ImGui::PopID();
    if (changed) {
        app.mapping_changed();
    }
}

static void draw_neutral_controls() {
    App& app = g_app;
    Config& cfg = app.config;
    const float em = ImGui::GetFontSize();
    const TrackerSnapshot* snap = app.snapshot.get();
    const bool face = snap && snap->face.found;

    ImGui::BeginDisabled(!face);
    push_button_colors(ImVec4(0.18f, 0.36f, 0.58f, 1));
    if (ImGui::Button("Set neutral", ImVec2(em * 9, ImGui::GetFrameHeight() * 1.5f))) {
        cfg.neutral = snap->face.rotation;
        cfg.neutral_set = true;
        app.mapping_changed();
    }
    ImGui::PopStyleColor(3);
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Hold your head in the position that should give no output, then click.");
    ImGui::SameLine();
    ImGui::BeginGroup();
    if (cfg.neutral_set) {
        ImGui::TextUnformatted("Neutral: set.");
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            cfg.neutral_set = false;
            cfg.neutral = Mat3();
            app.mapping_changed();
        }
        ImGui::SetItemTooltip("Go back to facing the camera as neutral.");
    } else {
        ImGui::TextUnformatted("Neutral: facing the camera.");
    }
    ImGui::TextDisabled(face ? "Hold your head where it should give no output and click."
                             : "Neutral can be set when a face is found.");
    ImGui::EndGroup();
}

// ---------------------------------------------------------------------------

static void draw_top_bar() {
    App& app = g_app;
    Config& cfg = app.config;
    const float em = ImGui::GetFontSize();

    const ImVec2 big(em * 9, ImGui::GetFrameHeight() * 1.7f);
    push_button_colors(app.running ? ImVec4(0.13f, 0.52f, 0.22f, 1) : ImVec4(0.45f, 0.17f, 0.17f, 1));
    if (ImGui::Button(app.running ? "RUNNING" : "STOPPED", big)) {
        app.running = !app.running;
    }
    ImGui::PopStyleColor(3);
    ImGui::SetItemTooltip(app.running ? "Click to stop: releases everything and stops sending inputs.\n"
                                        "The face is still tracked and the output still shown."
                                      : "Click to start sending inputs to the receiver.");

    ImGui::SameLine(0, em * 1.5f);
    ImGui::BeginGroup();
    const float label_w = em * 6;
    // Receiver
    static char ip_buffer[64] = {};
    static bool ip_init = false;
    if (!ip_init) {
        strncpy_s(ip_buffer, cfg.receiver_ip.c_str(), _TRUNCATE);
        ip_init = true;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Receiver IP");
    ImGui::SameLine(label_w);
    ImGui::SetNextItemWidth(em * 9);
    if (ImGui::InputTextWithHint("##ip", "e.g. 192.168.1.50", ip_buffer, sizeof(ip_buffer),
            ImGuiInputTextFlags_CharsNoBlank)) {
        cfg.receiver_ip = ip_buffer;
        app.sender.set_receiver(cfg.receiver_ip);
        app.mark_dirty();
    }
    ImGui::SetItemTooltip("IP address of the HID Forwarder receiver (Pico W). Inputs are sent as UDP to port 42734.");
    ImGui::SameLine();
    if (cfg.receiver_ip.empty()) {
        ImGui::TextDisabled("not set");
    } else if (!app.sender.address_valid()) {
        ImGui::TextColored(COLOR_ERROR, "invalid IPv4 address");
    } else if (app.sender.send_error()) {
        ImGui::TextColored(COLOR_ERROR, "send failed");
    } else if (app.running) {
        ImGui::TextColored(COLOR_OK, "sending (%llu packets)", (unsigned long long) app.sender.packets_sent());
    } else {
        ImGui::TextDisabled("idle");
    }

    // Camera
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Camera");
    ImGui::SameLine(label_w);
    ImGui::SetNextItemWidth(em * 16);
    std::string current = "Default (first camera)";
    for (const CameraDevice& d : app.cameras) {
        if (d.id == cfg.camera_id) {
            current = d.name;
        }
    }
    if (!cfg.camera_id.empty() && current == "Default (first camera)") {
        current = "(unavailable camera)";
    }
    if (ImGui::BeginCombo("##camera", current.c_str())) {
        if (ImGui::IsWindowAppearing()) {
            app.cameras = list_cameras();
        }
        if (ImGui::Selectable("Default (first camera)", cfg.camera_id.empty())) {
            cfg.camera_id.clear();
            app.camera.set_device(cfg.camera_id);
            app.mark_dirty();
        }
        for (const CameraDevice& d : app.cameras) {
            ImGui::PushID(d.id.c_str());
            if (ImGui::Selectable(d.name.c_str(), d.id == cfg.camera_id)) {
                cfg.camera_id = d.id;
                app.camera.set_device(cfg.camera_id);
                app.mark_dirty();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    const std::string capturing = app.camera.device_name();
    if (!capturing.empty()) {
        ImGui::SetItemTooltip("Capturing from: %s", capturing.c_str());
    }
    ImGui::SameLine();
    const std::string cam_error = app.camera.error();
    const TrackerSnapshot* snap = app.snapshot.get();
    if (!app.tracker_error.empty()) {
        ImGui::TextColored(COLOR_ERROR, "face tracking failed to start");
    } else if (!cam_error.empty() && capturing.empty()) {
        ImGui::TextColored(COLOR_ERROR, "%s", cam_error.c_str());
    } else if (!capturing.empty()) {
        const std::string format = app.camera.format();
        if (snap && snap->fps > 0) {
            ImGui::TextDisabled("%s, receiving %.0f fps, tracking %.1f ms per frame", format.c_str(), snap->fps,
                snap->process_ms);
        } else {
            ImGui::TextDisabled("%s, waiting for frames", format.c_str());
        }
    } else {
        ImGui::TextDisabled("opening...");
    }
    ImGui::EndGroup();
}

static void draw_ui() {
    App& app = g_app;
    Config& cfg = app.config;
    ImGuiIO& io = ImGui::GetIO();
    const float em = ImGui::GetFontSize();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("##main", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

    draw_top_bar();
    ImGui::Spacing();

    const float right_w = std::max(em * 24, ImGui::GetContentRegionAvail().x * 0.4f);
    const float left_w = ImGui::GetContentRegionAvail().x - right_w - ImGui::GetStyle().ItemSpacing.x;

    if (ImGui::BeginChild("left", ImVec2(left_w, 0))) {
        ImGui::SeparatorText("Preview");
        bool changed = false;
        changed |= ImGui::Checkbox("Privacy", &cfg.privacy);
        ImGui::SetItemTooltip("Show only the tracked face features, not the camera image.");
        ImGui::SameLine(0, em * 1.5f);
        changed |= ImGui::Checkbox("Zoom to face", &cfg.zoom_to_face);
        ImGui::SameLine(0, em * 1.5f);
        changed |= ImGui::Checkbox("Mirror", &cfg.mirror);
        ImGui::SetItemTooltip("Show the image like a mirror. Doesn't change the output.");
        if (changed) {
            app.mark_dirty();
        }
        const float below = ImGui::GetFrameHeight() * 1.5f + ImGui::GetStyle().ItemSpacing.y * 3 + em;
        draw_preview(ImGui::GetContentRegionAvail().y - below);
        ImGui::Spacing();
        draw_neutral_controls();
    }
    ImGui::EndChild();
    ImGui::SameLine();
    if (ImGui::BeginChild("right", ImVec2(0, 0))) {
        ImGui::SeparatorText("Head angles");
        for (int a = 0; a < HEAD_AXIS_COUNT; a++) {
            if (a > 0) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
            }
            draw_axis(a);
        }
        ImGui::Spacing();
        ImGui::SeparatorText("Smoothing");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (ImGui::SliderFloat("##smoothing", &cfg.smoothing, 0.0f, 1.0f, "%.2f")) {
            app.mapping_changed();
        }
        ImGui::SetItemTooltip("Higher = steadier output when holding still, but a little more lag.\n"
                              "Fast movements are smoothed less either way.");
        ImGui::SeparatorText("Output");
        draw_output();
    }
    ImGui::EndChild();

    ImGui::End();
}

// ---------------------------------------------------------------------------

static LRESULT WINAPI main_wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) {
        return true;
    }
    switch (msg) {
        case WM_SIZE:
            capture_window_placement();
            if (wparam == SIZE_MINIMIZED) {
                return 0;
            }
            g_resize_width = (UINT) LOWORD(lparam);
            g_resize_height = (UINT) HIWORD(lparam);
            return 0;
        case WM_MOVE:
            capture_window_placement();
            return 0;
        case WM_CLOSE:
            capture_window_placement();
            break;
        case WM_GETMINMAXINFO: {
            float s = GetDpiForWindow(hwnd) / 96.0f;
            auto* mmi = (MINMAXINFO*) lparam;
            mmi->ptMinTrackSize.x = (LONG) (900 * s);
            mmi->ptMinTrackSize.y = (LONG) (640 * s);
            return 0;
        }
        case WM_DPICHANGED: {
            g_pending_dpi_scale = HIWORD(wparam) / 96.0f;
            const RECT* r = (const RECT*) lparam;
            SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_SYSCOMMAND:
            if ((wparam & 0xfff0) == SC_KEYMENU) {  // disable ALT application menu
                return 0;
            }
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show_cmd) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    MFStartup(MF_VERSION, MFSTARTUP_LITE);

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    App& app = g_app;
    app.config_path = app_data_dir() + L"\\settings.json";
    load_config(app.config_path, app.config);
    if (wcsstr(GetCommandLineW(), L"--start") != nullptr) {
        app.running = true;
    }

    WNDCLASSEXW wc = { sizeof(wc) };
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = main_wnd_proc;
    wc.hInstance = instance;
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"FaceHidTransmitter";
    RegisterClassExW(&wc);

    // Restore the window where it was last time, but only if its title bar
    // would still be reachable (monitors may have changed since).
    const WindowRect& saved = app.config.window;
    bool restore = false;
    if (saved.valid) {
        RECT title = { saved.x, saved.y, saved.x + saved.w, saved.y + GetSystemMetrics(SM_CYCAPTION) };
        if (HMONITOR mon = MonitorFromRect(&title, MONITOR_DEFAULTTONULL)) {
            MONITORINFO mi = { sizeof(mi) };
            RECT visible;
            restore = GetMonitorInfoW(mon, &mi) && IntersectRect(&visible, &title, &mi.rcWork) &&
                      visible.right - visible.left >= 100;
        }
    }

    const float primary_scale = ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY));
    HWND hwnd = restore
        ? CreateWindowW(wc.lpszClassName, L"Face HID Transmitter", WS_OVERLAPPEDWINDOW,
              saved.x, saved.y, saved.w, saved.h, nullptr, nullptr, instance, nullptr)
        : CreateWindowW(wc.lpszClassName, L"Face HID Transmitter", WS_OVERLAPPEDWINDOW,
              (int) (80 * primary_scale), (int) (80 * primary_scale), (int) (1240 * primary_scale),
              (int) (820 * primary_scale), nullptr, nullptr, instance, nullptr);
    app.main_hwnd = hwnd;
    const float scale = GetDpiForWindow(hwnd) / 96.0f;

    if (!create_device(hwnd)) {
        cleanup_device();
        MessageBoxW(nullptr, L"Failed to initialize Direct3D 11.", L"Face HID Transmitter", MB_ICONERROR);
        return 1;
    }

    app.sender.set_receiver(app.config.receiver_ip);
    app.sender.start();
    app.mapping_changed();
    app.config_dirty = false;
    if (app.tracker.start(&app.sender, app.tracker_error)) {
        app.cameras = list_cameras();
        app.camera.set_device(app.config.camera_id);
        app.camera.start([](std::shared_ptr<const Image> frame) { g_app.tracker.push_frame(std::move(frame)); });
    }

    if (restore) {
        WINDOWPLACEMENT wp = { sizeof(wp) };
        GetWindowPlacement(hwnd, &wp);
        wp.rcNormalPosition = { saved.x, saved.y, saved.x + saved.w, saved.y + saved.h };
        wp.showCmd = show_cmd;
        if (saved.maximized && (show_cmd == SW_SHOWNORMAL || show_cmd == SW_SHOW || show_cmd == SW_SHOWDEFAULT)) {
            wp.showCmd = SW_SHOWMAXIMIZED;
        }
        SetWindowPlacement(hwnd, &wp);
    } else {
        ShowWindow(hwnd, show_cmd);
    }
    UpdateWindow(hwnd);
    g_track_placement = true;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    apply_style(scale);
    // Latin plus the degree sign.
    static const ImWchar glyph_ranges[] = { 0x0020, 0x00FF, 0 };
    io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 17.0f, nullptr, glyph_ranges);
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_device, g_device_context);

    bool done = false;
    while (!done) {
        const bool minimized = IsIconic(hwnd) != 0;
        if (minimized || g_swap_chain_occluded) {
            // Nothing to draw: sleep until something happens.
            MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
        }
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) {
                done = true;
            }
        }
        if (done) {
            break;
        }

        app.sender.set_active(app.running);
        if (app.config_dirty && GetTickCount64() - app.config_dirty_since >= 1000) {
            save_now();
        }

        if (minimized) {
            continue;
        }
        if (g_swap_chain_occluded && g_swap_chain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            continue;
        }
        g_swap_chain_occluded = false;

        if (g_resize_width != 0 && g_resize_height != 0) {
            cleanup_render_target();
            g_swap_chain->ResizeBuffers(0, g_resize_width, g_resize_height, DXGI_FORMAT_UNKNOWN, 0);
            g_resize_width = g_resize_height = 0;
            create_render_target();
        }
        if (g_pending_dpi_scale > 0) {
            apply_style(g_pending_dpi_scale);
            g_pending_dpi_scale = 0;
        }

        app.snapshot = app.tracker.snapshot();
        if (app.snapshot && app.snapshot->frame && app.snapshot->seq != app.uploaded_seq && !app.config.privacy) {
            upload_frame(*app.snapshot->frame);
            app.uploaded_seq = app.snapshot->seq;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        draw_ui();
        ImGui::Render();

        const float clear[4] = { 0.1f, 0.1f, 0.12f, 1.0f };
        g_device_context->OMSetRenderTargets(1, &g_render_target_view, nullptr);
        g_device_context->ClearRenderTargetView(g_render_target_view, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        HRESULT hr = g_swap_chain->Present(1, 0);
        g_swap_chain_occluded = (hr == DXGI_STATUS_OCCLUDED);
    }

    app.running = false;
    app.camera.stop();
    app.tracker.stop();
    app.sender.set_active(false);
    app.sender.stop();
    if (app.config_dirty) {
        save_now();
    }

    app.snapshot.reset();
    release_texture();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    cleanup_device();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    WSACleanup();
    MFShutdown();
    CoUninitialize();
    return 0;
}
