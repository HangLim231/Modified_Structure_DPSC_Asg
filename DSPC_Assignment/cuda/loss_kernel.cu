// File: cuda/loss_kernel.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
using namespace std;

// Kernel for cross-entropy loss calculation
__global__ void cross_entropy_loss_kernel(
    const float* predictions,  // [N x num_classes] output from network
    const int* targets,        // [N] target class indices
    float* losses,             // [N] per-sample loss values
    int N, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int target = targets[idx];
    float pred = predictions[idx * num_classes + target];

    // Clip prediction to avoid log(0)
    pred = fmaxf(pred, 1e-15f);
    losses[idx] = -logf(pred);
}

// Kernel for computing gradients of cross-entropy loss
__global__ void cross_entropy_grad_kernel(
    const float* predictions,  // [N x num_classes] output from network
    const int* targets,        // [N] target class indices
    float* gradients,          // [N x num_classes] gradient output
    int N, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int target = targets[idx];

    // Initialize all gradients to the softmax output
    for (int c = 0; c < num_classes; c++) {
        gradients[idx * num_classes + c] = predictions[idx * num_classes + c];

        // For target class, subtract 1
        if (c == target) {
            gradients[idx * num_classes + c] -= 1.0f;
        }
    }
}

// Wrapper to calculate cross-entropy loss
float compute_loss(
    const float* d_predictions,
    const int* d_targets,
    float* d_losses,
    int N, int num_classes)
{
    dim3 grid((N + 255) / 256);
    dim3 block(256);

    cross_entropy_loss_kernel << <grid, block >> > (
        d_predictions, d_targets, d_losses, N, num_classes);

    // Copy loss values back to host for averaging
    float* h_losses = new float[N];
    cudaMemcpy(h_losses, d_losses, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Calculate average loss
    float total_loss = 0.0f;
    for (int i = 0; i < N; i++) {
        total_loss += h_losses[i];
    }

    delete[] h_losses;
    return total_loss / N;
}

// Wrapper to compute cross-entropy loss gradients
void compute_loss_gradients(
    const float* d_predictions,
    const int* d_targets,
    float* d_gradients,
    int N, int num_classes)
{
    dim3 grid((N + 255) / 256);
    dim3 block(256);

    cross_entropy_grad_kernel << <grid, block >> > (
        d_predictions, d_targets, d_gradients, N, num_classes);
}