// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

// Windows-specific parts of the face tracker: models embedded as resources,
// and image loading with WIC for tests.

#include "face.h"

#include <windows.h>

#include <wincodec.h>

#include "resources.h"

// The module (exe or DLL) this code is linked into, which holds the models.
static HMODULE this_module() {
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR) &this_module, &module);
    return module;
}

static bool load_resource(int id, const uint8_t*& data, size_t& size) {
    const HMODULE module = this_module();
    HRSRC res = FindResourceW(module, MAKEINTRESOURCEW(id), (LPCWSTR) RT_RCDATA);
    if (res == nullptr) {
        return false;
    }
    HGLOBAL handle = LoadResource(module, res);
    if (handle == nullptr) {
        return false;
    }
    data = (const uint8_t*) LockResource(handle);
    size = SizeofResource(module, res);
    return data != nullptr;
}

bool FaceTracker::init(std::string& error) {
    FaceModels models;
    size_t size;
    if (!load_resource(IDR_FACE_DETECTION, models.detector, models.detector_size)) {
        error = "face detection model missing";
        return false;
    }
    if (!load_resource(IDR_FACE_LANDMARK, models.landmarker, models.landmarker_size)) {
        error = "face landmark model missing";
        return false;
    }
    const uint8_t* obj;
    if (!load_resource(IDR_CANONICAL_FACE, obj, size)) {
        error = "canonical face model missing";
        return false;
    }
    models.canonical_obj = (const char*) obj;
    models.canonical_obj_size = size;
    return init(models, error);
}

bool load_image_file(const std::wstring& path, Image& image) {
    IWICImagingFactory* factory = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) {
        return false;
    }
    bool ok = false;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    UINT w = 0, h = 0;
    if (SUCCEEDED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, &decoder)) &&
        SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(factory->CreateFormatConverter(&converter)) &&
        SUCCEEDED(converter->Initialize(frame, GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0,
            WICBitmapPaletteTypeCustom)) &&
        SUCCEEDED(converter->GetSize(&w, &h))) {
        image.width = (int) w;
        image.height = (int) h;
        image.pixels.resize((size_t) w * h * 4);
        ok = SUCCEEDED(converter->CopyPixels(nullptr, w * 4, (UINT) image.pixels.size(), image.pixels.data()));
    }
    if (converter) {
        converter->Release();
    }
    if (frame) {
        frame->Release();
    }
    if (decoder) {
        decoder->Release();
    }
    factory->Release();
    return ok;
}
