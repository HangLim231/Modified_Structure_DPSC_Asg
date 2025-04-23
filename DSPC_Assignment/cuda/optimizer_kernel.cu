// File: cuda/optimizer_kernel.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

// Kernel for SGD optimizer update with momentum
__global__ void sgd_update_kernel(
    float* params,           // Parameters to update (weights or biases)
    const float* gradients,  // Computed gradients for the parameters
    float* velocities,       // Momentum velocities
    float learning_rate,     // Learning rate
    float momentum,          // Momentum coefficient
    int size)                // Size of the parameter array
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Update velocity with momentum
    velocities[idx] = momentum * velocities[idx] - learning_rate * gradients[idx];

    // Update parameters
    params[idx] += velocities[idx];
}

// Wrapper for SGD optimizer update
void sgd_update(
    float* d_params,
    const float* d_gradients,
    float* d_velocities,
    float learning_rate,
    float momentum,
    int size)
{
    dim3 grid((size + 255) / 256);
    dim3 block(256);

    sgd_update_kernel << <grid, block >> > (
        d_params, d_gradients, d_velocities,
        learning_rate, momentum, size);

    cudaDeviceSynchronize();
}