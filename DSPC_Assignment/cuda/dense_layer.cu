// File: cuda/dense_layer.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
using namespace std;

// Kernel for dense layer forward pass
__global__ void dense_forward_kernel(
    const float* input,    // N x InputSize
    const float* weights,  // OutputSize x InputSize
    const float* bias,     // OutputSize
    float* output,         // N x OutputSize
    int N, int InputSize, int OutputSize)
{
    int n = blockIdx.y; // Batch dimension
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index

    if (out_idx >= OutputSize) return;

    float sum = bias[out_idx];

    for (int i = 0; i < InputSize; i++) {
        sum += input[n * InputSize + i] * weights[out_idx * InputSize + i];
    }

    output[n * OutputSize + out_idx] = fmaxf(0.0f, sum); // Hardcoded ReLU activation
}

// Kernel for dense layer forward pass
void dense_forward(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int N, int InputSize, int OutputSize)
{
    dim3 grid((OutputSize + 255) / 256, N); //Hardcoded fixed grid size
    dim3 block(256);    //Hardcoded fixed threads block size

    dense_forward_kernel << <grid, block >> > (
        d_input, d_weights, d_bias, d_output,
        N, InputSize, OutputSize
        );
    cudaDeviceSynchronize();
}