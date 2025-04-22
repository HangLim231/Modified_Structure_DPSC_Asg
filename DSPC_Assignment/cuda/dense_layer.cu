// File: cuda/dense_layer.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
using namespace std;

__global__ void dense_forward_kernel(
    const float* input,    // N x InputSize
    const float* weights,  // OutputSize x InputSize
    const float* bias,     // OutputSize
    float* output,         // N x OutputSize
    int N, int InputSize, int OutputSize)
{
    int n = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= OutputSize) return;

    float sum = bias[out_idx];

    for (int i = 0; i < InputSize; i++) {
        sum += input[n * InputSize + i] * weights[out_idx * InputSize + i];
    }

    output[n * OutputSize + out_idx] = fmaxf(0.0f, sum); // Apply ReLU
}

void dense_forward(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int N, int InputSize, int OutputSize)
{
    dim3 grid((OutputSize + 255) / 256, N);
    dim3 block(256);

    dense_forward_kernel << <grid, block >> > (
        d_input, d_weights, d_bias, d_output,
        N, InputSize, OutputSize
        );
    cudaDeviceSynchronize();
}