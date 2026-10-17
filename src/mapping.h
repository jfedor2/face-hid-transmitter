// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include "gamepad.h"

enum class HeadAxis : int {
    Yaw,
    Pitch,
    Roll,
    Count
};

constexpr int HEAD_AXIS_COUNT = (int) HeadAxis::Count;
extern const char* const HEAD_AXIS_NAMES[HEAD_AXIS_COUNT];         // "Yaw"
extern const char* const HEAD_AXIS_DESCRIPTIONS[HEAD_AXIS_COUNT];  // "turn left / right"

// What a head angle drives. For sticks, turning right / looking up / tilting
// right moves the stick right / up. Triggers: above neutral drives R2, below
// drives L2. L2 / R2 alone are driven by the positive side only.
enum class AxisTarget : int {
    None,
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    Triggers,
    L2,
    R2,
    Count
};

constexpr int AXIS_TARGET_COUNT = (int) AxisTarget::Count;
extern const char* const AXIS_TARGET_NAMES[AXIS_TARGET_COUNT];

struct AxisMapping {
    AxisTarget target = AxisTarget::None;
    bool invert = false;
    float range = 20.0f;    // degrees from neutral for full output
    float deadzone = 3.0f;  // degrees from neutral with no output
};

// The output (-1..1) for an angle relative to neutral, in degrees.
float axis_output(const AxisMapping& m, float angle);

// Adds the output of one axis to the gamepad state. Values add up when several
// axes drive the same output; clamp_state() caps them.
void apply_axis(const AxisMapping& m, float value, GamepadState& state);
void clamp_state(GamepadState& state);

// One Euro filter (Casiez et al.): smooths jitter when still, with little lag
// when moving.
class OneEuroFilter {
public:
    void reset() { initialized_ = false; }
    // min_cutoff in Hz: lower = smoother when still. beta: how fast the
    // cutoff rises with speed.
    float filter(float value, double t_seconds, float min_cutoff, float beta);

private:
    bool initialized_ = false;
    float x_ = 0.0f;
    float dx_ = 0.0f;
    double t_ = 0.0;
};
