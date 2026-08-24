// A CUDA program shaped like something worth watching.
//
// Every GPUFlow demo needs a program to point at, and a trivial one makes the
// tool look trivial. The launch pattern here imitates a transformer training
// step: a long projection GEMM on the compute stream, quick attention and
// normalisation kernels on a second, a parameter update at the end of each
// step, and host/device copies on a third. The point is that the three lanes
// end up with three visibly different rhythms — uniform kernels of equal
// length draw as a striped bar and tell a viewer nothing.
//
// The arithmetic is deliberately meaningless. This measures nothing and
// computes nothing; it exists to occupy the GPU in a legible pattern.
//
// This file is not part of the CMake build. GPUFlow itself needs only NVML to
// compile, and requiring nvcc for a demo would put a CUDA toolkit between a
// stranger and their first build.
//
//   nvcc -o demo_workload examples/demo_workload.cu -lnvToolsExt
//   ./build/gpuflow run -- ./demo_workload 600
//
// Then open http://127.0.0.1:7717 and zoom into the TRACED process.

#include <nvToolsExtCudaRt.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

// Iteration counts, not physics: these are tuned so the four kernels land at
// clearly different durations on a mid-range card. Adjust if your lanes come
// out looking uniform.
namespace {
constexpr int kProjectionWork = 42000;
constexpr int kSoftmaxWork = 5000;
constexpr int kNormWork = 9000;
constexpr int kUpdateWork = 18000;
constexpr int kMicroBatches = 24;
}  // namespace

__global__ void gemm_qkv_projection(float* out, int iters) {
    float a = threadIdx.x;
    for (int i = 0; i < iters; ++i) a = fmaf(a, 1.0000001f, 0.5f);
    out[threadIdx.x] = a;
}

__global__ void attention_softmax(float* out, int iters) {
    float a = threadIdx.x;
    for (int i = 0; i < iters; ++i) a = a * 0.9999f + 1.0f;
    out[threadIdx.x] = a;
}

__global__ void layernorm_forward(float* out, int iters) {
    float a = threadIdx.x;
    for (int i = 0; i < iters; ++i) a = fmaf(a, 0.999999f, 0.25f);
    out[threadIdx.x] = a;
}

__global__ void adam_step(float* out, int iters) {
    float a = threadIdx.x;
    for (int i = 0; i < iters; ++i) a = a * 0.99998f + 0.001f;
    out[threadIdx.x] = a;
}

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 600;

    float* activations = nullptr;
    float* attention = nullptr;
    float* staging = nullptr;
    float host[1024] = {};

    // Three streams because three lanes is the smallest number that shows the
    // plane is drawing structure rather than a single bar.
    cudaStream_t compute, side, copy;
    cudaStreamCreate(&compute);
    cudaStreamCreate(&side);
    cudaStreamCreate(&copy);

    // CUDA has no stream names of its own, so a profiler can only show numbers
    // unless the program says otherwise. These three calls are the whole reason
    // GPUFlow can label the lanes; without them it would honestly draw s13, s14
    // and s15, because it would honestly not know.
    nvtxNameCudaStreamA(compute, "compute");
    nvtxNameCudaStreamA(side, "attention");
    nvtxNameCudaStreamA(copy, "transfer");

    cudaMalloc(&activations, sizeof(host));
    cudaMalloc(&attention, sizeof(host));
    cudaMalloc(&staging, sizeof(host));

    std::printf("[demo] transformer-shaped workload on 3 streams, %d s\n", seconds);
    std::fflush(stdout);

    const time_t start = time(nullptr);
    long steps = 0;
    while (time(nullptr) - start < seconds) {
        for (int micro = 0; micro < kMicroBatches; ++micro) {
            gemm_qkv_projection<<<8, 256, 0, compute>>>(activations, kProjectionWork);
            attention_softmax<<<8, 256, 0, side>>>(attention, kSoftmaxWork);
            layernorm_forward<<<8, 256, 0, side>>>(attention, kNormWork);
        }
        // Once per step, so the compute lane gets a visible punctuation mark
        // between runs of the projection kernel.
        adam_step<<<8, 256, 0, compute>>>(activations, kUpdateWork);

        cudaMemcpyAsync(staging, host, sizeof(host), cudaMemcpyHostToDevice, copy);
        cudaMemcpyAsync(host, staging, sizeof(host), cudaMemcpyDeviceToHost, copy);

        cudaDeviceSynchronize();
        ++steps;
    }

    std::printf("[demo] %ld steps\n", steps);

    cudaFree(activations);
    cudaFree(attention);
    cudaFree(staging);
    cudaStreamDestroy(compute);
    cudaStreamDestroy(side);
    cudaStreamDestroy(copy);
    return 0;
}
