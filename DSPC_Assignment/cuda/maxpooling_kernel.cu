// File: cuda/maxpooling_kernel.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <cmath>
using namespace std;

// Kernel for 2D max pooling
__global__ void maxpool2d_forward_kernel(
    const float* input, // input feature map (N x C x H x W)
    float* output, // output feature map (N x C x H_out x W_out)
    int N, int C, int H, int W,
    int pool_size, int stride)
{
    int n = blockIdx.z; // Batch dimension
    int c = blockIdx.y; // Output channel dimension
    int hw = blockIdx.x * blockDim.x + threadIdx.x; // Spatial dimension

    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    if (hw >= H_out * W_out) return;

    int h_out = hw / W_out;
    int w_out = hw % W_out;

    float max_val = -FLT_MAX;

    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            int h_in = h_out * stride + i;
            int w_in = w_out * stride + j;
            if (h_in < H && w_in < W) {
                int input_idx = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    output[output_idx] = max_val;
}

// Kernel for 2D max pooling
void maxpool2d_forward(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    int pool_size, int stride)
{
    int H_out = (H - pool_size) / stride + 1; // No padding support
    int W_out = (W - pool_size) / stride + 1;
    dim3 grid((H_out * W_out + 255) / 256, C, N); // Hardcoded fixed grid size
    dim3 block(256); // Hardcoded fixed threads block size

    maxpool2d_forward_kernel << <grid, block >> > (
        d_input, d_output, N, C, H, W, pool_size, stride
        );
    cudaDeviceSynchronize();
}
