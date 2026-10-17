// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A minimal TensorFlow Lite interpreter, just enough for the MediaPipe face
// detection and face landmark models: float32 NHWC tensors, batch size 1, and
// the handful of ops those models use (CONV_2D, DEPTHWISE_CONV_2D, ADD, PAD,
// MAX_POOL_2D, RELU, PRELU, RESHAPE, CONCATENATION, DEQUANTIZE of float16
// constants).
class TfliteModel {
public:
    // Parses the model and prepares it for running. The data is only used
    // during the call. On failure, error() says why.
    bool load(const uint8_t* data, size_t size);
    const std::string& error() const { return error_; }

    // The input tensor (the first one), to be filled before run().
    float* input() { return tensors_[inputs_[0]].data.data(); }
    const std::vector<int>& input_shape() const { return tensors_[inputs_[0]].shape; }

    void run();

    size_t output_count() const { return outputs_.size(); }
    const float* output(size_t i) const { return tensors_[outputs_[i]].data.data(); }
    size_t output_size(size_t i) const { return tensors_[outputs_[i]].data.size(); }

private:
    struct Tensor {
        std::vector<int> shape;
        std::vector<float> data;
        bool constant = false;
        // H, W, C of a 4D NHWC tensor (or of a lower rank one, padded with 1s).
        int h() const { return dim(-3); }
        int w() const { return dim(-2); }
        int c() const { return dim(-1); }
        int dim(int from_end) const {
            int i = (int) shape.size() + from_end;
            return i >= 0 ? shape[i] : 1;
        }
    };

    enum class OpType {
        Conv,
        DepthwiseConv,
        Add,
        Pad,
        MaxPool,
        Relu,
        Prelu,
        Copy,  // RESHAPE
        Concat,
    };

    struct Op {
        OpType type;
        std::vector<int> in;
        int out = -1;
        // Conv, DepthwiseConv, MaxPool
        int stride_w = 1, stride_h = 1;
        int kw = 1, kh = 1;
        int pad_top = 0, pad_left = 0;
        int depth_multiplier = 1;
        int activation = 0;  // 0 = none, 1 = relu, 3 = relu6
        // Conv weights rearranged to [kh][kw][in_c][out_c],
        // depthwise to [kh][kw][out_c]
        std::vector<float> weights;
        std::vector<float> bias;
        // Pad: [before, after] for each of the 4 dimensions
        int pads[4][2] = {};
        // Concat
        int axis = 0;
    };

    bool fail(const std::string& message);
    void conv(const Op& op);
    void depthwise_conv(const Op& op);
    void add(const Op& op);
    void pad(const Op& op);
    void max_pool(const Op& op);
    void prelu(const Op& op);
    void concat(const Op& op);

    std::vector<Tensor> tensors_;
    std::vector<Op> ops_;
    std::vector<int> inputs_;
    std::vector<int> outputs_;
    std::string error_;
};
