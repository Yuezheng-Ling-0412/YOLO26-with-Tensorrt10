cmake_minimum_required(VERSION 3.18)
project(USBcamera LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin Nano Super (Ampere)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ── TensorRT ──────────────────────────────────────────────────────────────────
set(TRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include")
set(TRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu"     CACHE PATH "TensorRT lib")
include_directories(${TRT_INCLUDE_DIR})
find_library(TRT_LIB     nvinfer        PATHS ${TRT_LIB_DIR} REQUIRED)
find_library(TRT_ONNX    nvonnxparser   PATHS ${TRT_LIB_DIR} REQUIRED)

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── Threads ───────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
set(CXX_SOURCES
    src/main.cpp
    src/trt_engine.cpp
    src/postprocessor.cpp
)

set(CUDA_SOURCES
    src/cuda_preprocessor.cu
)

include_directories(include)

add_executable(main ${CXX_SOURCES} ${CUDA_SOURCES})

target_link_libraries(main
    ${TRT_LIB}
    ${TRT_ONNX}
    ${CUDA_LIBRARIES}
    cuda          # libcuda.so  (cudaHostRegister 等)
    ${OpenCV_LIBS}
    Threads::Threads
)

