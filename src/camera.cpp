// Copyright (c) 2026 Jacek Fedorynski
// SPDX-License-Identifier: MIT

#include "camera.h"

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include <cmath>
#include <cstring>

#include "util.h"

// A camera that hasn't sent a frame for this long is reopened.
static constexpr ULONGLONG FRAME_TIMEOUT_MS = 5000;
static constexpr DWORD RETRY_MS = 2000;
static constexpr DWORD STREAM = (DWORD) MF_SOURCE_READER_FIRST_VIDEO_STREAM;

template <typename T>
static void release(T*& p) {
    if (p != nullptr) {
        p->Release();
        p = nullptr;
    }
}

static std::string allocated_string(IMFActivate* a, const GUID& key) {
    WCHAR* s = nullptr;
    UINT32 len = 0;
    std::string out;
    if (SUCCEEDED(a->GetAllocatedString(key, &s, &len))) {
        out = to_utf8(s);
        CoTaskMemFree(s);
    }
    return out;
}

std::vector<CameraDevice> list_cameras() {
    std::vector<CameraDevice> out;
    if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) {
        return out;
    }
    IMFAttributes* attr = nullptr;
    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    if (SUCCEEDED(MFCreateAttributes(&attr, 1)) &&
        SUCCEEDED(attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)) &&
        SUCCEEDED(MFEnumDeviceSources(attr, &devices, &count))) {
        for (UINT32 i = 0; i < count; i++) {
            CameraDevice d;
            d.name = allocated_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
            d.id = allocated_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
            if (!d.id.empty()) {
                out.push_back(d);
            }
            devices[i]->Release();
        }
        CoTaskMemFree(devices);
    }
    release(attr);
    MFShutdown();
    return out;
}

static std::string hresult_message(HRESULT hr) {
    if (hr == E_ACCESSDENIED) {
        return "Access denied. Allow desktop apps to use the camera in Windows privacy settings.";
    }
    if (hr == MF_E_HW_MFT_FAILED_START_STREAMING || hr == MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED ||
        hr == MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED || hr == HRESULT_FROM_WIN32(ERROR_SHARING_VIOLATION)) {
        return "The camera is in use by another app.";
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "Camera error 0x%08lx.", (unsigned long) hr);
    return buf;
}

// Receives the frames from the source reader (in asynchronous mode, so it
// can always be stopped; a synchronous ReadSample() can block forever).
class ReaderCallback : public IMFSourceReaderCallback {
public:
    explicit ReaderCallback(Camera* camera) : camera_(camera) {
        done_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        flush_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }

    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) override {
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IMFSourceReaderCallback)) {
            *ppv = static_cast<IMFSourceReaderCallback*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&refs_); }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG n = InterlockedDecrement(&refs_);
        if (n == 0) {
            delete this;
        }
        return n;
    }

    STDMETHODIMP OnReadSample(HRESULT hr, DWORD, DWORD flags, LONGLONG, IMFSample* sample) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) {
            return S_OK;
        }
        if (SUCCEEDED(hr) && (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR))) {
            hr = E_FAIL;
        }
        if (SUCCEEDED(hr) && (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) && !read_format()) {
            hr = E_FAIL;
        }
        if (SUCCEEDED(hr) && sample != nullptr) {
            deliver(sample);
        }
        if (SUCCEEDED(hr)) {
            hr = reader->ReadSample(STREAM, 0, nullptr, nullptr, nullptr, nullptr);
        }
        if (FAILED(hr)) {
            status = hr;
            SetEvent(done_event);
        }
        return S_OK;
    }

    STDMETHODIMP OnFlush(DWORD) override {
        SetEvent(flush_event);
        return S_OK;
    }

    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent*) override { return S_OK; }

    // Reads the output size and stride from the reader.
    bool read_format() {
        IMFMediaType* current = nullptr;
        if (FAILED(reader->GetCurrentMediaType(STREAM, &current))) {
            return false;
        }
        UINT32 w = 0, h = 0, stride = 0, num = 0, den = 1;
        MFGetAttributeSize(current, MF_MT_FRAME_SIZE, &w, &h);
        MFGetAttributeRatio(current, MF_MT_FRAME_RATE, &num, &den);
        width = w;
        height = h;
        default_stride = SUCCEEDED(current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride)) ? (LONG) stride : (LONG) w * 4;
        current->Release();
        char buf[64];
        snprintf(buf, sizeof(buf), "%ux%u, %.0f fps", w, h, den ? (double) num / den : 0.0);
        camera_->set_format(buf);
        return w > 0 && h > 0;
    }

    // Everything below is guarded by mutex (except the events).
    std::mutex mutex;
    IMFSourceReader* reader = nullptr;  // not owned
    bool stopping = false;
    HRESULT status = S_OK;
    bool got_frame = false;
    ULONGLONG last_frame_ms = 0;
    UINT32 width = 0, height = 0;
    LONG default_stride = 0;
    HANDLE done_event;   // set on an error
    HANDLE flush_event;  // set when a flush is done

private:
    ~ReaderCallback() {
        CloseHandle(done_event);
        CloseHandle(flush_event);
    }

    void deliver(IMFSample* sample) {
        IMFMediaBuffer* buffer = nullptr;
        if (FAILED(sample->ConvertToContiguousBuffer(&buffer))) {
            return;
        }
        auto frame = std::make_shared<Image>();
        frame->width = (int) width;
        frame->height = (int) height;
        frame->pixels.resize((size_t) width * height * 4);
        const size_t row = (size_t) width * 4;
        IMF2DBuffer* buffer2d = nullptr;
        BYTE* scan0 = nullptr;
        LONG pitch = 0;
        bool ok = false;
        if (SUCCEEDED(buffer->QueryInterface(IID_PPV_ARGS(&buffer2d))) && SUCCEEDED(buffer2d->Lock2D(&scan0, &pitch))) {
            for (UINT32 y = 0; y < height; y++) {
                memcpy(&frame->pixels[y * row], scan0 + (LONG_PTR) y * pitch, row);
            }
            buffer2d->Unlock2D();
            ok = true;
        } else {
            BYTE* data = nullptr;
            DWORD len = 0;
            if (SUCCEEDED(buffer->Lock(&data, nullptr, &len))) {
                const LONG abs_stride = std::abs(default_stride);
                if ((size_t) len >= (size_t) abs_stride * height && abs_stride >= (LONG) row) {
                    for (UINT32 y = 0; y < height; y++) {
                        const UINT32 src_row = default_stride < 0 ? height - 1 - y : y;
                        memcpy(&frame->pixels[y * row], data + (size_t) src_row * abs_stride, row);
                    }
                    ok = true;
                }
                buffer->Unlock();
            }
        }
        release(buffer2d);
        release(buffer);
        if (ok) {
            got_frame = true;
            last_frame_ms = GetTickCount64();
            camera_->sink_(std::move(frame));
        }
    }

    Camera* camera_;
    ULONG refs_ = 1;
};

void Camera::start(Sink sink) {
    sink_ = std::move(sink);
    quit_ = false;
    wake_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    thread_ = std::thread(&Camera::thread_main, this);
}

void Camera::stop() {
    quit_ = true;
    if (wake_event_ != nullptr) {
        SetEvent(wake_event_);
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    if (wake_event_ != nullptr) {
        CloseHandle(wake_event_);
        wake_event_ = nullptr;
    }
}

void Camera::set_device(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id == requested_id_) {
        return;
    }
    requested_id_ = id;
    device_changed_ = true;
    if (wake_event_ != nullptr) {
        SetEvent(wake_event_);
    }
}

std::string Camera::device_name() {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_name_;
}

std::string Camera::format() {
    std::lock_guard<std::mutex> lock(mutex_);
    return format_;
}

std::string Camera::error() {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void Camera::set_error(const std::string& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = e;
}

void Camera::set_format(const std::string& f) {
    std::lock_guard<std::mutex> lock(mutex_);
    format_ = f;
}

void Camera::thread_main() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFStartup(MF_VERSION, MFSTARTUP_LITE);
    while (!quit_) {
        std::string id;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            id = requested_id_;
            device_changed_ = false;
        }
        run_device(id);
        // Try again in a bit, or right away if the device was changed.
        if (!quit_ && !device_changed_) {
            WaitForSingleObject((HANDLE) wake_event_, RETRY_MS);
        }
    }
    MFShutdown();
    CoUninitialize();
}

// Prefers 30 fps or more, then a size close to 1280x720.
static double format_score(UINT32 w, UINT32 h, double fps) {
    double score = std::min(fps, 30.0) * 100;
    score -= std::abs((double) w * h - 1280.0 * 720) / 10000;
    return score;
}

// Opens the camera and reads frames until it fails, quit or the device changes.
void Camera::run_device(const std::string& requested) {
    std::vector<CameraDevice> cameras = list_cameras();
    const CameraDevice* dev = nullptr;
    for (const CameraDevice& c : cameras) {
        if (requested.empty() || c.id == requested) {
            dev = &c;
            break;
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        device_name_.clear();
        format_.clear();
    }
    if (dev == nullptr) {
        set_error(cameras.empty() ? "No camera found." : "The selected camera is not connected.");
        return;
    }

    ReaderCallback* callback = new ReaderCallback(this);
    IMFAttributes* attr = nullptr;
    IMFMediaSource* source = nullptr;
    IMFAttributes* reader_attr = nullptr;
    IMFSourceReader* reader = nullptr;
    IMFMediaType* best_type = nullptr;
    IMFMediaType* out_type = nullptr;
    HRESULT hr = MFCreateAttributes(&attr, 2);
    if (SUCCEEDED(hr)) {
        hr = attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    }
    if (SUCCEEDED(hr)) {
        hr = attr->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, from_utf8(dev->id).c_str());
    }
    if (SUCCEEDED(hr)) {
        hr = MFCreateDeviceSource(attr, &source);
    }
    if (SUCCEEDED(hr)) {
        hr = MFCreateAttributes(&reader_attr, 2);
    }
    if (SUCCEEDED(hr)) {
        // Converts whatever the camera gives (YUY2, NV12, MJPG) to RGB32.
        hr = reader_attr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    }
    if (SUCCEEDED(hr)) {
        hr = reader_attr->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, callback);
    }
    if (SUCCEEDED(hr)) {
        hr = MFCreateSourceReaderFromMediaSource(source, reader_attr, &reader);
    }
    if (SUCCEEDED(hr)) {
        double best = -1e18;
        for (DWORD i = 0;; i++) {
            IMFMediaType* t = nullptr;
            if (FAILED(reader->GetNativeMediaType(STREAM, i, &t))) {
                break;
            }
            UINT32 w = 0, h = 0, num = 0, den = 1;
            MFGetAttributeSize(t, MF_MT_FRAME_SIZE, &w, &h);
            MFGetAttributeRatio(t, MF_MT_FRAME_RATE, &num, &den);
            const double score = format_score(w, h, den ? (double) num / den : 0);
            if (w > 0 && h > 0 && score > best) {
                best = score;
                release(best_type);
                best_type = t;
            } else {
                t->Release();
            }
        }
        if (best_type != nullptr) {
            reader->SetCurrentMediaType(STREAM, nullptr, best_type);
        }
        hr = MFCreateMediaType(&out_type);
    }
    if (SUCCEEDED(hr)) {
        out_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        hr = reader->SetCurrentMediaType(STREAM, nullptr, out_type);
    }
    if (SUCCEEDED(hr)) {
        std::lock_guard<std::mutex> lock(callback->mutex);
        callback->reader = reader;
        if (!callback->read_format()) {
            hr = E_FAIL;
        }
    }
    if (SUCCEEDED(hr)) {
        hr = reader->ReadSample(STREAM, 0, nullptr, nullptr, nullptr, nullptr);
    }

    if (SUCCEEDED(hr)) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            device_name_ = dev->name;
            error_.clear();
        }
        const ULONGLONG opened = GetTickCount64();
        const HANDLE events[2] = { (HANDLE) wake_event_, callback->done_event };
        while (!quit_ && !device_changed_) {
            const DWORD r = WaitForMultipleObjects(2, events, FALSE, 1000);
            std::lock_guard<std::mutex> lock(callback->mutex);
            if (r == WAIT_OBJECT_0 + 1) {
                set_error(callback->got_frame ? "The camera stopped sending frames." : hresult_message(callback->status));
                break;
            }
            const ULONGLONG since = GetTickCount64() - (callback->got_frame ? callback->last_frame_ms : opened);
            if (since > FRAME_TIMEOUT_MS) {
                set_error(callback->got_frame ? "The camera stopped sending frames."
                                              : "The camera isn't sending frames. Is another app using it?");
                break;
            }
        }
    } else if (!quit_ && !device_changed_) {
        set_error(hresult_message(hr));
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        device_name_.clear();
        format_.clear();
    }
    if (reader != nullptr) {
        {
            std::lock_guard<std::mutex> lock(callback->mutex);
            callback->stopping = true;
        }
        // Cancels a pending ReadSample.
        if (SUCCEEDED(reader->Flush(STREAM))) {
            WaitForSingleObject(callback->flush_event, 2000);
        }
    }
    release(out_type);
    release(best_type);
    release(reader);
    release(reader_attr);
    if (source != nullptr) {
        source->Shutdown();
    }
    release(source);
    release(attr);
    callback->Release();
}
