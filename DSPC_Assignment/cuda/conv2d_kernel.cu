// File: cuda/conv2d_kernel.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
using namespace std;

__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int n = blockIdx.z;
    int co = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    if (hw >= H_out * W_out) return;

    int h = hw / W_out;
    int w = hw % W_out;

    float sum = bias[co];

    for (int ci = 0; ci < C_in; ci++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int in_h = h + p;
                int in_w = w + q;
                int input_idx = n * (C_in * H * W) + ci * (H * W) + in_h * W + in_w;
                int weight_idx = co * (C_in * K * K) + ci * (K * K) + p * K + q;
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    int out_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w;
    output[out_idx] = fmaxf(0.0f, sum);
}

void conv2d_forward(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    dim3 grid((H_out * W_out + 255) / 256, C_out, N);
    dim3 block(256);

    conv2d_forward_kernel <<<grid, block>>> (
        d_input, d_weights, d_bias, d_output,
        N, C_in, H, W, C_out, K
        );
    cudaDeviceSynchronize();
}
