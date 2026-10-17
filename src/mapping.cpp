// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "mapping.h"

#include <algorithm>
#include <cmath>

const char* const HEAD_AXIS_NAMES[HEAD_AXIS_COUNT] = {
    "Yaw",
    "Pitch",
    "Roll",
};

const char* const HEAD_AXIS_DESCRIPTIONS[HEAD_AXIS_COUNT] = {
    "turn left / right",
    "look down / up",
    "tilt left / right",
};

const char* const AXIS_TARGET_NAMES[AXIS_TARGET_COUNT] = {
    "None",
    "Left stick X",
    "Left stick Y",
    "Right stick X",
    "Right stick Y",
    "L2 / R2",
    "L2",
    "R2",
};

float axis_output(const AxisMapping& m, float angle) {
    if (m.invert) {
        angle = -angle;
    }
    const float mag = std::abs(angle);
    if (mag <= m.deadzone) {
        return 0.0f;
    }
    const float span = std::max(m.range - m.deadzone, 0.1f);
    const float out = std::min((mag - m.deadzone) / span, 1.0f);
    return angle < 0 ? -out : out;
}

void apply_axis(const AxisMapping& m, float v, GamepadState& state) {
    float(&ls)[2] = state.sticks[(int) Stick::Left];
    float(&rs)[2] = state.sticks[(int) Stick::Right];
    float& l2 = state.triggers[(int) Trigger::L2];
    float& r2 = state.triggers[(int) Trigger::R2];
    switch (m.target) {
        case AxisTarget::None:
        case AxisTarget::Count:
            break;
        case AxisTarget::LeftStickX:
            ls[0] += v;
            break;
        case AxisTarget::LeftStickY:
            ls[1] -= v;  // -1 = up
            break;
        case AxisTarget::RightStickX:
            rs[0] += v;
            break;
        case AxisTarget::RightStickY:
            rs[1] -= v;
            break;
        case AxisTarget::Triggers:
            (v > 0 ? r2 : l2) += std::abs(v);
            break;
        case AxisTarget::L2:
            l2 += std::max(v, 0.0f);
            break;
        case AxisTarget::R2:
            r2 += std::max(v, 0.0f);
            break;
    }
}

void clamp_state(GamepadState& state) {
    for (auto& s : state.sticks) {
        for (float& a : s) {
            a = std::clamp(a, -1.0f, 1.0f);
        }
    }
    for (float& t : state.triggers) {
        t = std::clamp(t, 0.0f, 1.0f);
    }
}

static float smoothing_alpha(float cutoff, double dt) {
    const double tau = 1.0 / (2 * 3.14159265358979 * cutoff);
    return (float) (1.0 / (1.0 + tau / dt));
}

float OneEuroFilter::filter(float value, double t, float min_cutoff, float beta) {
    if (!initialized_ || t <= t_) {
        if (!initialized_) {
            x_ = value;
            dx_ = 0.0f;
        }
        initialized_ = true;
        t_ = t;
        return x_;
    }
    const double dt = std::min(t - t_, 0.5);
    t_ = t;
    const float dx = (value - x_) / (float) dt;
    dx_ += smoothing_alpha(1.0f, dt) * (dx - dx_);
    const float cutoff = min_cutoff + beta * std::abs(dx_);
    x_ += smoothing_alpha(cutoff, dt) * (value - x_);
    return x_;
}
