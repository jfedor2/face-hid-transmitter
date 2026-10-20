cmake_minimum_required(VERSION 3.20)
project(face-hid-transmitter CXX RC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# static CRT so the exe runs without the VC++ redistributable
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(FetchContent)
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.9b
  GIT_SHALLOW TRUE
)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(imgui json)

# MediaPipe face models (Apache 2.0), embedded in the exe as resources.
set(MODEL_DIR ${CMAKE_BINARY_DIR}/models)
function(download_model name url sha256)
  if(NOT EXISTS ${MODEL_DIR}/${name})
    file(DOWNLOAD ${url} ${MODEL_DIR}/${name} EXPECTED_HASH SHA256=${sha256} STATUS status)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
      file(REMOVE ${MODEL_DIR}/${name})
      message(FATAL_ERROR "Downloading ${url} failed: ${status}")
    endif()
  endif()
endfunction()
download_model(face_detection_short_range.tflite
  https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
  3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633)
download_model(face_landmark.tflite
  https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
  1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814)
download_model(canonical_face_model.obj
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.14/mediapipe/modules/face_geometry/data/canonical_face_model.obj
  8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)

# Everything except the UI, so the test tool can link it too.
add_library(fht_core STATIC
  src/tflite.cpp
  src/face.cpp
  src/face_win.cpp
  src/camera.cpp
  src/mapping.cpp
  src/sender.cpp
  src/tracker.cpp
  src/gamepad.cpp
  src/config.cpp
)
target_compile_definitions(fht_core PUBLIC UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(fht_core PUBLIC /W3 /utf-8)
target_link_libraries(fht_core PUBLIC
  nlohmann_json::nlohmann_json
  ws2_32 ole32 mf mfplat mfreadwrite mfuuid windowscodecs
)

# The models are compiled into each exe that uses them.
set(MODEL_RC ${CMAKE_BINARY_DIR}/models.rc)
file(WRITE ${MODEL_RC}.in
  "#include \"${CMAKE_SOURCE_DIR}/src/resources.h\"\n"
  "IDR_FACE_DETECTION RCDATA \"${MODEL_DIR}/face_detection_short_range.tflite\"\n"
  "IDR_FACE_LANDMARK RCDATA \"${MODEL_DIR}/face_landmark.tflite\"\n"
  "IDR_CANONICAL_FACE RCDATA \"${MODEL_DIR}/canonical_face_model.obj\"\n")
configure_file(${MODEL_RC}.in ${MODEL_RC} COPYONLY)

add_executable(face-hid-transmitter WIN32 src/main.cpp ${MODEL_RC})
target_link_libraries(face-hid-transmitter PRIVATE fht_core imgui_lib d3d11 dxgi d3dcompiler shell32)

# Command line tool for running the models and the tracker on files.
add_executable(fht-test src/test_main.cpp ${MODEL_RC})
target_link_libraries(fht-test PRIVATE fht_core shell32)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT face-hid-transmitter)
