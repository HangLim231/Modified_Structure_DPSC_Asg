// File: cuda/backprop_kernel.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
using namespace std;

// Kernel for dense layer backward pass (calculating gradients for weights)
__global__ void dense_backward_weights_kernel(
    const float* input,         // [N x InputSize]
    const float* grad_output,   // [N x OutputSize]
    float* grad_weights,        // [OutputSize x InputSize]
    int N, int InputSize, int OutputSize)
{
    int out_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= OutputSize || in_idx >= InputSize) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        bool activation_active = (input[n * InputSize + in_idx] > 0); // ReLU derivative
        sum += grad_output[n * OutputSize + out_idx] * activation_active * input[n * InputSize + in_idx];
    }

    grad_weights[out_idx * InputSize + in_idx] = sum / N;
}

// Kernel for dense layer backward pass (calculating gradients for biases)
__global__ void dense_backward_bias_kernel(
    const float* grad_output,   // [N x OutputSize]
    float* grad_bias,           // [OutputSize]
    int N, int OutputSize)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= OutputSize) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += grad_output[n * OutputSize + out_idx];
    }

    grad_bias[out_idx] = sum / N;
}

// Kernel for dense layer backward pass (calculating gradients for inputs)
__global__ void dense_backward_input_kernel(
    const float* weights,       // [OutputSize x InputSize]
    const float* grad_output,   // [N x OutputSize]
    float* grad_input,          // [N x InputSize]
    int N, int InputSize, int OutputSize)
{
    int n = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || in_idx >= InputSize) return;

    float sum = 0.0f;
    for (int out_idx = 0; out_idx < OutputSize; out_idx++) {
        sum += grad_output[n * OutputSize + out_idx] * weights[out_idx * InputSize + in_idx];
    }

    // Apply ReLU gradient
    grad_input[n * InputSize + in_idx] = sum;
}

// Wrapper for dense layer backward pass
void dense_backward(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int N, int InputSize, int OutputSize)
{
    // Calculate gradients for weights
    dim3 grid_weights(InputSize + 255 / 256, OutputSize);
    dim3 block_weights(256);
    dense_backward_weights_kernel << <grid_weights, block_weights >> > (
        d_input, d_grad_output, d_grad_weights, N, InputSize, OutputSize);

    // Calculate gradients for biases
    dim3 grid_bias((OutputSize + 255) / 256);
    dim3 block_bias(256);
    dense_backward_bias_kernel << <grid_bias, block_bias >> > (
        d_grad_output, d_grad_bias, N, OutputSize);

    // Calculate gradients for inputs
    dim3 grid_input((InputSize + 255) / 256, N);
    dim3 block_input(256);
    dense_backward_input_kernel << <grid_input, block_input >> > (
        d_weights, d_grad_output, d_grad_input, N, InputSize, OutputSize);

    cudaDeviceSynchronize();
}

// Kernel for convolution backward pass (calculating gradients for weights)
__global__ void conv2d_backward_weights_kernel(
    const float* input,       // [N x C_in x H x W]
    const float* grad_output, // [N x C_out x H_out x W_out]
    float* grad_weights,      // [C_out x C_in x K x K]
    int N, int C_in, int H, int W,
    int C_out, int K, int H_out, int W_out)
{
    int co = blockIdx.z;
    int ci = blockIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (co >= C_out || ci >= C_in || k_idx >= K * K) return;

    int kh = k_idx / K;
    int kw = k_idx % K;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int in_h = h + kh;
                int in_w = w + kw;
                int grad_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w;
                int in_idx = n * (C_in * H * W) + ci * (H * W) + in_h * W + in_w;

                sum += grad_output[grad_idx] * input[in_idx];
            }
        }
    }

    int weight_idx = co * (C_in * K * K) + ci * (K * K) + kh * K + kw;
    grad_weights[weight_idx] = sum / N;
}

// Kernel for convolution backward pass (calculating gradients for biases)
__global__ void conv2d_backward_bias_kernel(
    const float* grad_output, // [N x C_out x H_out x W_out]
    float* grad_bias,         // [C_out]
    int N, int C_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;

    if (co >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w;
                sum += grad_output[idx];
            }
        }
    }

    grad_bias[co] = sum / N;
}

// Kernel for convolution backward pass (calculating gradients for inputs)
__global__ void conv2d_backward_input_kernel(
    const float* weights,     // [C_out x C_in x K x K]
    const float* grad_output, // [N x C_out x H_out x W_out]
    float* grad_input,        // [N x C_in x H x W]
    int N, int C_in, int H, int W,
    int C_out, int K, int H_out, int W_out)
{
    int n = blockIdx.z;
    int ci = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || ci >= C_in || hw >= H * W) return;

    int h = hw / W;
    int w = hw % W;

    float sum = 0.0f;
    for (int co = 0; co < C_out; co++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int h_out = h - kh;
                int w_out = w - kw;

                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int grad_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + h_out * W_out + w_out;
                    int weight_idx = co * (C_in * K * K) + ci * (K * K) + (K - 1 - kh) * K + (K - 1 - kw);

                    sum += grad_output[grad_idx] * weights[weight_idx];
                }
            }
        }
    }

    int in_idx = n * (C_in * H * W) + ci * (H * W) + h * W + w;
    grad_input[in_idx] = sum;
}

// Wrapper for convolution backward pass
void conv2d_backward(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Calculate gradients for weights
    dim3 grid_weights((K * K + 255) / 256, C_in, C_out);
    dim3 block_weights(256);
    conv2d_backward_weights_kernel << <grid_weights, block_weights >> > (
        d_input, d_grad_output, d_grad_weights,
        N, C_in, H, W, C_out, K, H_out, W_out);

    // Calculate gradients for biases
    dim3 grid_bias((C_out + 255) / 256);
    dim3 block_bias(256);
    conv2d_backward_bias_kernel << <grid_bias, block_bias >> > (
        d_grad_output, d_grad_bias, N, C_out, H_out, W_out);

    // Calculate gradients for inputs
    dim3 grid_input((H * W + 255) / 256, C_in, N);
    dim3 block_input(256);
    conv2d_backward_input_kernel << <grid_input, block_input >> > (
        d_weights, d_grad_output, d_grad_input,
        N, C_in, H, W, C_out, K, H_out, W_out);

    cudaDeviceSynchronize();
}

// Kernel for max pooling backward pass
__global__ void maxpool2d_backward_kernel(
    const float* input,       // [N x C x H x W] original input to maxpool
    const float* output,      // [N x C x H_out x W_out] output from maxpool
    const float* grad_output, // [N x C x H_out x W_out] gradient of output
    float* grad_input,        // [N x C x H x W] gradient of input
    int N, int C, int H, int W,
    int pool_size, int stride)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;

    if (n >= N || c >= C || hw >= H_out * W_out) return;

    int h_out = hw / W_out;
    int w_out = hw % W_out;

    int h_start = h_out * stride;
    int w_start = w_out * stride;

    int h_end = min(h_start + pool_size, H);
    int w_end = min(w_start + pool_size, W);

    // Get the output value to find the corresponding input index
    int out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    float out_val = output[out_idx];
    float grad_val = grad_output[out_idx];

    // Find the position in the input that contributed to the output
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            int in_idx = n * (C * H * W) + c * (H * W) + h * W + w;
            if (input[in_idx] == out_val) {
                // This is the element that was selected by the max pooling
                atomicAdd(&grad_input[in_idx], grad_val);
                // Only one element in each pooling window contributes to the gradient
                return;
            }
        }
    }
}

// Wrapper for max pooling backward pass
void maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_grad_output,
    float* d_grad_input,
    int N, int C, int H, int W,
    int pool_size, int stride)
{
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;

    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, sizeof(float) * N * C * H * W);

    dim3 grid((H_out * W_out + 255) / 256, C, N);
    dim3 block(256);

    maxpool2d_backward_kernel << <grid, block >> > (
        d_input, d_output, d_grad_output, d_grad_input,
        N, C, H, W, pool_size, stride);

    cudaDeviceSynchronize();
}