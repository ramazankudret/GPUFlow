cmake_minimum_required(VERSION 3.16)
project(gpuflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(gpuflow
    src/main.cpp
    src/model/snapshot.cpp
    src/nvml/device_monitor.cpp
    src/transport/sse_server.cpp)

target_include_directories(gpuflow PRIVATE src)
target_compile_options(gpuflow PRIVATE -Wall -Wextra -Wpedantic)

# Prefer CMake's own CUDAToolkit module, which defines CUDA::nvml when a CUDA
# installation is discoverable. Fall back to a direct search for boxes that
# carry only the driver plus libnvidia-ml-dev — a perfectly ordinary setup, and
# the one a deployment host is most likely to have.
find_package(CUDAToolkit QUIET)
if(TARGET CUDA::nvml)
    target_link_libraries(gpuflow PRIVATE CUDA::nvml)
else()
    find_path(NVML_INCLUDE_DIR nvml.h)
    find_library(NVML_LIBRARY NAMES nvidia-ml)
    if(NOT NVML_INCLUDE_DIR OR NOT NVML_LIBRARY)
        message(FATAL_ERROR
            "gpuflow: NVML not found. Install the NVIDIA driver (libnvidia-ml.so.1) "
            "and headers — on Debian/Ubuntu that is `libnvidia-ml-dev`, or the CUDA "
            "toolkit.")
    endif()
    target_include_directories(gpuflow PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(gpuflow PRIVATE ${NVML_LIBRARY})
endif()

# device_monitor.cpp resolves the running-process entry points at runtime, so
# the link has to keep dlsym available.
target_link_libraries(gpuflow PRIVATE ${CMAKE_DL_LIBS})
