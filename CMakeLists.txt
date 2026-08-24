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
    src/transport/sse_server.cpp
    src/ipc/shared_ring.cpp
    src/collector/run_session.cpp
    src/collector/trace_accumulator.cpp)

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
# the link has to keep dlsym available. run_session needs shm_open, which older
# glibc keeps in librt.
target_link_libraries(gpuflow PRIVATE ${CMAKE_DL_LIBS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(gpuflow PRIVATE rt)
endif()

# The event ring the v0.2 collector will write into. Proven standalone here,
# before any CUPTI code exists — a missed fence is far cheaper to find with a
# synthetic writer than from inside a library injected into someone else's
# CUDA process. Also the seed of the benchmark harness: --mode throughput
# reports the per-record cost the collector will impose.
add_executable(ring_stress
    tools/ring_stress.cpp
    src/ipc/shared_ring.cpp)
target_include_directories(ring_stress PRIVATE src)
target_compile_options(ring_stress PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(ring_stress PRIVATE Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(ring_stress PRIVATE rt)   # shm_open on older glibc
endif()

option(GPUFLOW_RING_DEBUG "Trap concurrent producers on the SPSC ring" OFF)
if(GPUFLOW_RING_DEBUG)
    target_compile_definitions(ring_stress PRIVATE GPUFLOW_RING_DEBUG)
endif()

# ---------------------------------------------------------------- collector
#
# CUPTI ships with the CUDA toolkit, not the driver, so a machine can run
# gpuflow's passive layer without ever being able to build this. That is the
# normal case on a deployment host, and it must not fail the build — the agent
# is useful without the collector, just not as interesting.
find_path(CUPTI_INCLUDE_DIR cupti.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
          /usr/local/cuda/extras/CUPTI/include
          /usr/local/cuda/targets/x86_64-linux/include
    PATH_SUFFIXES extras/CUPTI/include)
find_library(CUPTI_LIBRARY cupti
    HINTS /usr/local/cuda/extras/CUPTI/lib64
          /usr/local/cuda/targets/x86_64-linux/lib
    PATH_SUFFIXES extras/CUPTI/lib64)

if(CUPTI_INCLUDE_DIR AND CUPTI_LIBRARY)
    message(STATUS "gpuflow: CUPTI found — building the injected collector")

    add_library(gpuflow_inject SHARED src/collector/injection.cpp src/ipc/shared_ring.cpp)
    # The name the driver loads is the one in CUDA_INJECTION64_PATH; keeping it
    # literal makes the env var readable in a ps listing.
    set_target_properties(gpuflow_inject PROPERTIES OUTPUT_NAME gpuflow_inject PREFIX "lib")
    target_include_directories(gpuflow_inject PRIVATE src ${CUPTI_INCLUDE_DIR})
    target_compile_options(gpuflow_inject PRIVATE -Wall -Wextra -Wpedantic -fPIC)
    target_link_libraries(gpuflow_inject PRIVATE ${CUPTI_LIBRARY} ${CMAKE_DL_LIBS})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(gpuflow_inject PRIVATE rt)
    endif()

    add_executable(collector_probe tools/collector_probe.cpp src/ipc/shared_ring.cpp)
    target_include_directories(collector_probe PRIVATE src)
    target_compile_options(collector_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(collector_probe PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(collector_probe PRIVATE rt)
    endif()
    add_dependencies(collector_probe gpuflow_inject)
else()
    message(STATUS "gpuflow: CUPTI not found — skipping the collector. The agent still "
                   "builds; only `gpuflow run` needs it.")
endif()
