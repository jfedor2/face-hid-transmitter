// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "tflite.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// FlatBuffers reading, just what the TFLite schema needs.

namespace {

struct FlatBuffer {
    const uint8_t* data;
    size_t size;

    template <typename T>
    T read(size_t pos) const {
        if (pos + sizeof(T) > size || pos + sizeof(T) < pos) {
            throw std::runtime_error("model file truncated");
        }
        T v;
        memcpy(&v, data + pos, sizeof(T));
        return v;
    }
};

struct Vector;

struct Table {
    const FlatBuffer* fb = nullptr;
    size_t pos = 0;

    // Position of a field, 0 if it's not present.
    size_t field(int id) const {
        const size_t vtable = pos - fb->read<int32_t>(pos);
        const uint16_t vtable_size = fb->read<uint16_t>(vtable);
        if (4 + 2 * (size_t) id >= vtable_size) {
            return 0;
        }
        const uint16_t offset = fb->read<uint16_t>(vtable + 4 + 2 * id);
        return offset ? pos + offset : 0;
    }

    template <typename T>
    T get(int id, T def) const {
        size_t p = field(id);
        return p ? fb->read<T>(p) : def;
    }

    bool has(int id) const { return field(id) != 0; }

    Table table(int id) const;
    Vector vector(int id) const;
};

struct Vector {
    const FlatBuffer* fb = nullptr;
    size_t pos = 0;  // of the elements
    uint32_t length = 0;

    template <typename T>
    T at(uint32_t i) const {
        return fb->read<T>(pos + (size_t) i * sizeof(T));
    }

    Table table(uint32_t i) const {
        const size_t p = pos + (size_t) i * 4;
        return Table{ fb, p + fb->read<uint32_t>(p) };
    }
};

Table Table::table(int id) const {
    size_t p = field(id);
    if (!p) {
        return Table{ fb, 0 };
    }
    return Table{ fb, p + fb->read<uint32_t>(p) };
}

Vector Table::vector(int id) const {
    size_t p = field(id);
    if (!p) {
        return Vector{ fb, 0, 0 };
    }
    const size_t v = p + fb->read<uint32_t>(p);
    return Vector{ fb, v + 4, fb->read<uint32_t>(v) };
}

// schema.fbs enums
enum BuiltinOp {
    OP_ADD = 0,
    OP_CONCATENATION = 2,
    OP_CONV_2D = 3,
    OP_DEPTHWISE_CONV_2D = 4,
    OP_DEQUANTIZE = 6,
    OP_MAX_POOL_2D = 17,
    OP_RELU = 19,
    OP_RESHAPE = 22,
    OP_PAD = 34,
    OP_PRELU = 54,
};

enum TensorType {
    TYPE_FLOAT32 = 0,
    TYPE_FLOAT16 = 1,
    TYPE_INT32 = 2,
};

enum Padding {
    PADDING_SAME = 0,
    PADDING_VALID = 1,
};

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else if (exp != 0) {
        bits = sign | ((uint32_t) (exp + 112) << 23) | (mant << 13);
    } else if (mant == 0) {
        bits = sign;
    } else {
        // subnormal
        exp = 113;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        bits = sign | ((uint32_t) exp << 23) | ((mant & 0x3ff) << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

size_t element_count(const std::vector<int>& shape) {
    size_t n = 1;
    for (int d : shape) {
        n *= (size_t) d;
    }
    return n;
}

int same_padding(int in, int out, int stride, int kernel) {
    return std::max((out - 1) * stride + kernel - in, 0) / 2;
}

}  // namespace

// ---------------------------------------------------------------------------

bool TfliteModel::fail(const std::string& message) {
    error_ = message;
    tensors_.clear();
    ops_.clear();
    return false;
}

bool TfliteModel::load(const uint8_t* data, size_t size) {
    tensors_.clear();
    ops_.clear();
    inputs_.clear();
    outputs_.clear();
    error_.clear();
    try {
        FlatBuffer fb{ data, size };
        const Table model{ &fb, fb.read<uint32_t>(0) };

        std::vector<int> opcodes;
        const Vector codes = model.vector(1);
        for (uint32_t i = 0; i < codes.length; i++) {
            const Table c = codes.table(i);
            opcodes.push_back(std::max<int>(c.get<int8_t>(0, 0), c.get<int32_t>(3, 0)));
        }

        const Vector buffers = model.vector(4);
        const Vector subgraphs = model.vector(2);
        if (subgraphs.length < 1) {
            return fail("no subgraph");
        }
        const Table graph = subgraphs.table(0);

        const Vector tensors = graph.vector(0);
        tensors_.resize(tensors.length);
        for (uint32_t i = 0; i < tensors.length; i++) {
            const Table t = tensors.table(i);
            Tensor& tensor = tensors_[i];
            const Vector shape = t.vector(0);
            for (uint32_t d = 0; d < shape.length; d++) {
                tensor.shape.push_back(shape.at<int32_t>(d));
            }
            const int type = t.get<int8_t>(1, TYPE_FLOAT32);
            const uint32_t buffer = t.get<uint32_t>(2, 0);
            const size_t n = element_count(tensor.shape);
            Vector bytes{ &fb, 0, 0 };
            if (buffer != 0 && buffer < buffers.length) {
                bytes = buffers.table(buffer).vector(0);
            }
            if (bytes.length == 0) {
                if (type != TYPE_FLOAT32) {
                    return fail("unsupported non-float tensor " + std::to_string(i));
                }
                tensor.data.resize(n);
                continue;
            }
            tensor.constant = true;
            tensor.data.resize(n);
            const size_t elem = type == TYPE_FLOAT16 ? 2 : 4;
            if (bytes.length != n * elem) {
                return fail("constant tensor " + std::to_string(i) + " has the wrong size");
            }
            fb.read<uint8_t>(bytes.pos + bytes.length - 1);  // bounds check
            const uint8_t* p = data + bytes.pos;
            for (size_t k = 0; k < n; k++) {
                if (type == TYPE_FLOAT32) {
                    memcpy(&tensor.data[k], p + k * 4, 4);
                } else if (type == TYPE_FLOAT16) {
                    uint16_t h;
                    memcpy(&h, p + k * 2, 2);
                    tensor.data[k] = half_to_float(h);
                } else if (type == TYPE_INT32) {
                    int32_t v;
                    memcpy(&v, p + k * 4, 4);
                    tensor.data[k] = (float) v;
                } else {
                    return fail("unsupported constant type " + std::to_string(type));
                }
            }
        }

        auto tensor_list = [&](const Vector& v, std::vector<int>& out) {
            for (uint32_t i = 0; i < v.length; i++) {
                const int t = v.at<int32_t>(i);
                if (t < 0 || t >= (int) tensors_.size()) {
                    throw std::runtime_error("bad tensor index");
                }
                out.push_back(t);
            }
        };
        tensor_list(graph.vector(1), inputs_);
        tensor_list(graph.vector(2), outputs_);
        if (inputs_.empty() || outputs_.empty()) {
            return fail("no inputs or outputs");
        }

        const Vector operators = graph.vector(3);
        for (uint32_t i = 0; i < operators.length; i++) {
            const Table o = operators.table(i);
            const uint32_t opcode_index = o.get<uint32_t>(0, 0);
            if (opcode_index >= opcodes.size()) {
                return fail("bad opcode index");
            }
            const int code = opcodes[opcode_index];
            Op op;
            std::vector<int> outs;
            tensor_list(o.vector(1), op.in);
            tensor_list(o.vector(2), outs);
            if (outs.size() != 1 || op.in.empty()) {
                return fail("unsupported operator arity");
            }
            op.out = outs[0];
            Tensor& out = tensors_[op.out];
            const Tensor& in = tensors_[op.in[0]];
            const Table options = o.table(4);
            const std::string where = " (operator " + std::to_string(i) + ")";

            switch (code) {
                case OP_DEQUANTIZE:
                    // float16 weights, converted when loading
                    if (!in.constant) {
                        return fail("DEQUANTIZE of a non-constant" + where);
                    }
                    out.data = in.data;
                    out.constant = true;
                    continue;

                case OP_CONV_2D:
                case OP_DEPTHWISE_CONV_2D: {
                    const bool depthwise = code == OP_DEPTHWISE_CONV_2D;
                    if (op.in.size() < 3 || !tensors_[op.in[1]].constant || !tensors_[op.in[2]].constant) {
                        return fail("convolution without constant weights" + where);
                    }
                    const Tensor& filter = tensors_[op.in[1]];
                    op.type = depthwise ? OpType::DepthwiseConv : OpType::Conv;
                    const int padding = options.pos ? options.get<int8_t>(0, PADDING_SAME) : PADDING_SAME;
                    op.stride_w = options.pos ? options.get<int32_t>(1, 1) : 1;
                    op.stride_h = options.pos ? options.get<int32_t>(2, 1) : 1;
                    int dilation_w = 1, dilation_h = 1;
                    if (depthwise) {
                        op.depth_multiplier = options.pos ? options.get<int32_t>(3, 1) : 1;
                        op.activation = options.pos ? options.get<int8_t>(4, 0) : 0;
                        dilation_w = options.pos ? options.get<int32_t>(5, 1) : 1;
                        dilation_h = options.pos ? options.get<int32_t>(6, 1) : 1;
                    } else {
                        op.activation = options.pos ? options.get<int8_t>(3, 0) : 0;
                        dilation_w = options.pos ? options.get<int32_t>(4, 1) : 1;
                        dilation_h = options.pos ? options.get<int32_t>(5, 1) : 1;
                    }
                    if (dilation_w != 1 || dilation_h != 1 || op.depth_multiplier != 1) {
                        return fail("unsupported dilation or depth multiplier" + where);
                    }
                    if (filter.shape.size() != 4 || in.shape.size() != 4 || out.shape.size() != 4) {
                        return fail("convolution with non-4D tensors" + where);
                    }
                    op.kh = filter.shape[1];
                    op.kw = filter.shape[2];
                    const int oc = out.c();
                    const int ic = in.c();
                    if (filter.shape[depthwise ? 3 : 0] != oc || (depthwise ? oc != ic : filter.shape[3] != ic) ||
                        tensors_[op.in[2]].data.size() != (size_t) oc) {
                        return fail("convolution shape mismatch" + where);
                    }
                    if (padding == PADDING_SAME) {
                        op.pad_top = same_padding(in.h(), out.h(), op.stride_h, op.kh);
                        op.pad_left = same_padding(in.w(), out.w(), op.stride_w, op.kw);
                    }
                    op.bias = tensors_[op.in[2]].data;
                    if (depthwise) {
                        op.weights = filter.data;  // already [1][kh][kw][oc]
                    } else {
                        // [oc][kh][kw][ic] -> [kh][kw][ic][oc]
                        op.weights.resize(filter.data.size());
                        for (int o2 = 0; o2 < oc; o2++) {
                            for (int k = 0; k < op.kh * op.kw; k++) {
                                for (int i2 = 0; i2 < ic; i2++) {
                                    op.weights[((size_t) k * ic + i2) * oc + o2] =
                                        filter.data[((size_t) o2 * op.kh * op.kw + k) * ic + i2];
                                }
                            }
                        }
                    }
                    op.in.resize(1);
                    break;
                }

                case OP_MAX_POOL_2D: {
                    op.type = OpType::MaxPool;
                    const int padding = options.get<int8_t>(0, PADDING_SAME);
                    op.stride_w = options.get<int32_t>(1, 1);
                    op.stride_h = options.get<int32_t>(2, 1);
                    op.kw = options.get<int32_t>(3, 1);
                    op.kh = options.get<int32_t>(4, 1);
                    op.activation = options.get<int8_t>(5, 0);
                    if (in.shape.size() != 4 || out.shape.size() != 4 || in.c() != out.c()) {
                        return fail("max pool shape mismatch" + where);
                    }
                    if (padding == PADDING_SAME) {
                        op.pad_top = same_padding(in.h(), out.h(), op.stride_h, op.kh);
                        op.pad_left = same_padding(in.w(), out.w(), op.stride_w, op.kw);
                    }
                    break;
                }

                case OP_ADD:
                    op.type = OpType::Add;
                    op.activation = options.pos ? options.get<int8_t>(0, 0) : 0;
                    if (op.in.size() != 2 || tensors_[op.in[1]].data.size() != in.data.size() ||
                        out.data.size() != in.data.size()) {
                        return fail("ADD with broadcasting is not supported" + where);
                    }
                    break;

                case OP_RELU:
                    op.type = OpType::Relu;
                    if (out.data.size() != in.data.size()) {
                        return fail("RELU shape mismatch" + where);
                    }
                    break;

                case OP_PRELU: {
                    op.type = OpType::Prelu;
                    if (op.in.size() != 2 || !tensors_[op.in[1]].constant) {
                        return fail("PRELU without constant alpha" + where);
                    }
                    op.weights = tensors_[op.in[1]].data;
                    if (op.weights.size() != (size_t) in.c() || out.data.size() != in.data.size()) {
                        return fail("PRELU alpha must be per channel" + where);
                    }
                    op.in.resize(1);
                    break;
                }

                case OP_RESHAPE:
                    op.type = OpType::Copy;
                    if (out.data.size() != in.data.size()) {
                        return fail("RESHAPE size mismatch" + where);
                    }
                    op.in.resize(1);
                    break;

                case OP_PAD: {
                    op.type = OpType::Pad;
                    if (op.in.size() < 2 || !tensors_[op.in[1]].constant) {
                        return fail("PAD without constant paddings" + where);
                    }
                    const Tensor& paddings = tensors_[op.in[1]];
                    const int rank = (int) in.shape.size();
                    if (rank > 4 || paddings.data.size() != (size_t) rank * 2 || out.shape.size() != in.shape.size()) {
                        return fail("PAD shape mismatch" + where);
                    }
                    for (int d = 0; d < rank; d++) {
                        op.pads[4 - rank + d][0] = (int) paddings.data[d * 2];
                        op.pads[4 - rank + d][1] = (int) paddings.data[d * 2 + 1];
                        if (in.shape[d] + op.pads[4 - rank + d][0] + op.pads[4 - rank + d][1] != out.shape[d]) {
                            return fail("PAD shape mismatch" + where);
                        }
                    }
                    if (op.pads[0][0] || op.pads[0][1]) {
                        return fail("PAD of the batch dimension" + where);
                    }
                    op.in.resize(1);
                    break;
                }

                case OP_CONCATENATION: {
                    op.type = OpType::Concat;
                    op.axis = options.get<int32_t>(0, 0);
                    op.activation = options.get<int8_t>(1, 0);
                    const int rank = (int) out.shape.size();
                    if (op.axis < 0) {
                        op.axis += rank;
                    }
                    size_t total = 0;
                    for (int t : op.in) {
                        if (tensors_[t].shape.size() != (size_t) rank) {
                            return fail("CONCATENATION rank mismatch" + where);
                        }
                        total += tensors_[t].data.size();
                    }
                    if (op.axis < 0 || op.axis >= rank || total != out.data.size()) {
                        return fail("CONCATENATION shape mismatch" + where);
                    }
                    break;
                }

                default:
                    return fail("unsupported operator " + std::to_string(code) + where);
            }
            if (op.activation != 0 && op.activation != 1 && op.activation != 3) {
                return fail("unsupported activation" + where);
            }
            for (int t : op.in) {
                if (tensors_[t].constant && op.type != OpType::Add && op.type != OpType::Concat) {
                    return fail("unexpected constant input" + where);
                }
            }
            ops_.push_back(std::move(op));
        }
        for (int t : outputs_) {
            if (tensors_[t].constant) {
                return fail("constant output");
            }
        }
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    // Only the weights of constant tensors were needed; they're copied into the ops.
    for (Tensor& t : tensors_) {
        if (t.constant) {
            bool used = false;
            for (const Op& op : ops_) {
                used |= std::find(op.in.begin(), op.in.end(), &t - tensors_.data()) != op.in.end();
            }
            if (!used) {
                t.data = std::vector<float>();
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------

static inline void activate(float* __restrict p, int n, int activation) {
    if (activation == 1) {
        for (int i = 0; i < n; i++) {
            p[i] = p[i] > 0.0f ? p[i] : 0.0f;
        }
    } else if (activation == 3) {
        for (int i = 0; i < n; i++) {
            p[i] = std::min(std::max(p[i], 0.0f), 6.0f);
        }
    }
}

void TfliteModel::conv(const Op& op) {
    const Tensor& in = tensors_[op.in[0]];
    Tensor& out = tensors_[op.out];
    const int ih = in.h(), iw = in.w(), ic = in.c();
    const int oh = out.h(), ow = out.w(), oc = out.c();
    const float* __restrict src = in.data.data();
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            float* __restrict acc = &out.data[((size_t) oy * ow + ox) * oc];
            memcpy(acc, op.bias.data(), sizeof(float) * oc);
            for (int ky = 0; ky < op.kh; ky++) {
                const int iy = oy * op.stride_h - op.pad_top + ky;
                if (iy < 0 || iy >= ih) {
                    continue;
                }
                for (int kx = 0; kx < op.kw; kx++) {
                    const int ix = ox * op.stride_w - op.pad_left + kx;
                    if (ix < 0 || ix >= iw) {
                        continue;
                    }
                    const float* __restrict s = src + ((size_t) iy * iw + ix) * ic;
                    const float* __restrict w = &op.weights[(size_t) (ky * op.kw + kx) * ic * oc];
                    for (int c = 0; c < ic; c++) {
                        const float a = s[c];
                        const float* __restrict wr = w + (size_t) c * oc;
                        for (int o = 0; o < oc; o++) {
                            acc[o] += a * wr[o];
                        }
                    }
                }
            }
            activate(acc, oc, op.activation);
        }
    }
}

void TfliteModel::depthwise_conv(const Op& op) {
    const Tensor& in = tensors_[op.in[0]];
    Tensor& out = tensors_[op.out];
    const int ih = in.h(), iw = in.w();
    const int oh = out.h(), ow = out.w(), ch = out.c();
    const float* __restrict src = in.data.data();
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            float* __restrict acc = &out.data[((size_t) oy * ow + ox) * ch];
            memcpy(acc, op.bias.data(), sizeof(float) * ch);
            for (int ky = 0; ky < op.kh; ky++) {
                const int iy = oy * op.stride_h - op.pad_top + ky;
                if (iy < 0 || iy >= ih) {
                    continue;
                }
                for (int kx = 0; kx < op.kw; kx++) {
                    const int ix = ox * op.stride_w - op.pad_left + kx;
                    if (ix < 0 || ix >= iw) {
                        continue;
                    }
                    const float* __restrict s = src + ((size_t) iy * iw + ix) * ch;
                    const float* __restrict w = &op.weights[(size_t) (ky * op.kw + kx) * ch];
                    for (int c = 0; c < ch; c++) {
                        acc[c] += s[c] * w[c];
                    }
                }
            }
            activate(acc, ch, op.activation);
        }
    }
}

void TfliteModel::add(const Op& op) {
    const float* __restrict a = tensors_[op.in[0]].data.data();
    const float* __restrict b = tensors_[op.in[1]].data.data();
    Tensor& out = tensors_[op.out];
    float* __restrict o = out.data.data();
    const int n = (int) out.data.size();
    for (int i = 0; i < n; i++) {
        o[i] = a[i] + b[i];
    }
    activate(o, n, op.activation);
}

void TfliteModel::pad(const Op& op) {
    const Tensor& in = tensors_[op.in[0]];
    Tensor& out = tensors_[op.out];
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    const int ih = in.h(), iw = in.w(), ic = in.c();
    const int ow = out.w(), oc = out.c();
    for (int y = 0; y < ih; y++) {
        for (int x = 0; x < iw; x++) {
            memcpy(&out.data[(((size_t) y + op.pads[1][0]) * ow + x + op.pads[2][0]) * oc + op.pads[3][0]],
                &in.data[((size_t) y * iw + x) * ic], sizeof(float) * ic);
        }
    }
}

void TfliteModel::max_pool(const Op& op) {
    const Tensor& in = tensors_[op.in[0]];
    Tensor& out = tensors_[op.out];
    const int ih = in.h(), iw = in.w();
    const int oh = out.h(), ow = out.w(), ch = out.c();
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            float* __restrict acc = &out.data[((size_t) oy * ow + ox) * ch];
            std::fill(acc, acc + ch, -3.4e38f);
            for (int ky = 0; ky < op.kh; ky++) {
                const int iy = oy * op.stride_h - op.pad_top + ky;
                if (iy < 0 || iy >= ih) {
                    continue;
                }
                for (int kx = 0; kx < op.kw; kx++) {
                    const int ix = ox * op.stride_w - op.pad_left + kx;
                    if (ix < 0 || ix >= iw) {
                        continue;
                    }
                    const float* __restrict s = &in.data[((size_t) iy * iw + ix) * ch];
                    for (int c = 0; c < ch; c++) {
                        acc[c] = std::max(acc[c], s[c]);
                    }
                }
            }
            activate(acc, ch, op.activation);
        }
    }
}

void TfliteModel::prelu(const Op& op) {
    const Tensor& in = tensors_[op.in[0]];
    Tensor& out = tensors_[op.out];
    const int ch = (int) op.weights.size();
    const size_t pixels = in.data.size() / ch;
    const float* __restrict alpha = op.weights.data();
    for (size_t p = 0; p < pixels; p++) {
        const float* __restrict s = &in.data[p * ch];
        float* __restrict o = &out.data[p * ch];
        for (int c = 0; c < ch; c++) {
            o[c] = s[c] >= 0.0f ? s[c] : s[c] * alpha[c];
        }
    }
}

void TfliteModel::concat(const Op& op) {
    Tensor& out = tensors_[op.out];
    size_t outer = 1;
    for (int d = 0; d < op.axis; d++) {
        outer *= out.shape[d];
    }
    float* o = out.data.data();
    for (size_t i = 0; i < outer; i++) {
        for (int t : op.in) {
            const Tensor& in = tensors_[t];
            const size_t inner = in.data.size() / outer;
            memcpy(o, &in.data[i * inner], sizeof(float) * inner);
            o += inner;
        }
    }
    activate(out.data.data(), (int) out.data.size(), op.activation);
}

void TfliteModel::run() {
    for (const Op& op : ops_) {
        switch (op.type) {
            case OpType::Conv:
                conv(op);
                break;
            case OpType::DepthwiseConv:
                depthwise_conv(op);
                break;
            case OpType::Add:
                add(op);
                break;
            case OpType::Pad:
                pad(op);
                break;
            case OpType::MaxPool:
                max_pool(op);
                break;
            case OpType::Relu: {
                const Tensor& in = tensors_[op.in[0]];
                Tensor& out = tensors_[op.out];
                out.data = in.data;
                activate(out.data.data(), (int) out.data.size(), 1);
                break;
            }
            case OpType::Prelu:
                prelu(op);
                break;
            case OpType::Copy:
                tensors_[op.out].data = tensors_[op.in[0]].data;
                break;
            case OpType::Concat:
                concat(op);
                break;
        }
    }
}
