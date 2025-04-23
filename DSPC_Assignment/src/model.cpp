// File: src/model.cpp

#include "../include/model.h"
#include "../include/timer.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "../include/optimizer.h"
#include "../include/backprop.h"
using namespace std;
using namespace layers;

Model::Model() : d_weights1(nullptr), d_bias1(nullptr),
d_weights2(nullptr), d_bias2(nullptr),
d_weights3(nullptr), d_bias3(nullptr),
d_fc_weights(nullptr), d_fc_bias(nullptr) {}

// Helper function for weight initialization with proper scaling
void initializeWeights(float* d_weights, size_t size, float scale) {
    float* h_weights = new float[size];
    for (int i = 0; i < size; i++) {
        // He initialization (scaled for ReLU)
        h_weights[i] = (rand() / static_cast<float>(RAND_MAX) - 0.5f) * scale;
    }
    cudaMemcpy(d_weights, h_weights, sizeof(float) * size, cudaMemcpyHostToDevice);
    delete[] h_weights;
}

// Initialize model with additional buffers for training
void Model::initialize() {
    cout << "Initializing model weights and biases..." << endl;

    // Hardcoded network architecture sizes
    // Conv1: 1 input channel, 32 output channels, 3x3 kernel
    size_t weights1_size = 32 * 1 * 3 * 3;
    cudaMalloc(&d_weights1, sizeof(float) * weights1_size);
    cudaMalloc(&d_bias1, sizeof(float) * 32);

    // Conv2: 32 input channels, 64 output channels, 3x3 kernel
    size_t weights2_size = 64 * 32 * 3 * 3;
    cudaMalloc(&d_weights2, sizeof(float) * weights2_size);
    cudaMalloc(&d_bias2, sizeof(float) * 64);

    // Conv3: 64 input channels, 64 output channels, 3x3 kernel
    size_t weights3_size = 64 * 64 * 3 * 3;
    cudaMalloc(&d_weights3, sizeof(float) * weights3_size);
    cudaMalloc(&d_bias3, sizeof(float) * 64);

    // Fully connected: 64*4*4 input features, 10 output classes
    size_t fc_weights_size = 10 * (64 * 4 * 4); // After 3 convs + 2 pooling, final feature map is 4x4
    cudaMalloc(&d_fc_weights, sizeof(float) * fc_weights_size);
    cudaMalloc(&d_fc_bias, sizeof(float) * 10);

    // Initialize weights with scaled random values
    initializeWeights(d_weights1, weights1_size, sqrt(2.0f / (3 * 3)));
    initializeWeights(d_weights2, weights2_size, sqrt(2.0f / (3 * 3 * 32)));
    initializeWeights(d_weights3, weights3_size, sqrt(2.0f / (3 * 3 * 64)));
    initializeWeights(d_fc_weights, fc_weights_size, sqrt(2.0f / (64 * 4 * 4)));

    // Bias initialization to small positive values (to help ReLU units)
    float* h_bias = new float[32];
    for (int i = 0; i < 32; i++) h_bias[i] = 0.01f;
    cudaMemcpy(d_bias1, h_bias, sizeof(float) * 32, cudaMemcpyHostToDevice);

    delete[] h_bias;
    h_bias = new float[64];
    for (int i = 0; i < 64; i++) h_bias[i] = 0.01f;
    cudaMemcpy(d_bias2, h_bias, sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias3, h_bias, sizeof(float) * 64, cudaMemcpyHostToDevice);
    delete[] h_bias;

    h_bias = new float[10];
    for (int i = 0; i < 10; i++) h_bias[i] = 0.01f;
    cudaMemcpy(d_fc_bias, h_bias, sizeof(float) * 10, cudaMemcpyHostToDevice);
    delete[] h_bias;

    // Initialize velocity buffers for momentum
    cudaMalloc(&d_weights1_velocity, sizeof(float) * weights1_size);
    cudaMalloc(&d_bias1_velocity, sizeof(float) * 32);
    cudaMalloc(&d_weights2_velocity, sizeof(float) * weights2_size);
    cudaMalloc(&d_bias2_velocity, sizeof(float) * 64);
    cudaMalloc(&d_weights3_velocity, sizeof(float) * weights3_size);
    cudaMalloc(&d_bias3_velocity, sizeof(float) * 64);
    cudaMalloc(&d_fc_weights_velocity, sizeof(float) * fc_weights_size);
    cudaMalloc(&d_fc_bias_velocity, sizeof(float) * 10);

    // Zero-initialize velocity buffers
    cudaMemset(d_weights1_velocity, 0, sizeof(float) * weights1_size);
    cudaMemset(d_bias1_velocity, 0, sizeof(float) * 32);
    cudaMemset(d_weights2_velocity, 0, sizeof(float) * weights2_size);
    cudaMemset(d_bias2_velocity, 0, sizeof(float) * 64);
    cudaMemset(d_weights3_velocity, 0, sizeof(float) * weights3_size);
    cudaMemset(d_bias3_velocity, 0, sizeof(float) * 64);
    cudaMemset(d_fc_weights_velocity, 0, sizeof(float) * fc_weights_size);
    cudaMemset(d_fc_bias_velocity, 0, sizeof(float) * 10);

    // Allocate gradient buffers
    cudaMalloc(&d_weights1_grad, sizeof(float) * weights1_size);
    cudaMalloc(&d_bias1_grad, sizeof(float) * 32);
    cudaMalloc(&d_weights2_grad, sizeof(float) * weights2_size);
    cudaMalloc(&d_bias2_grad, sizeof(float) * 64);
    cudaMalloc(&d_weights3_grad, sizeof(float) * weights3_size);
    cudaMalloc(&d_bias3_grad, sizeof(float) * 64);
    cudaMalloc(&d_fc_weights_grad, sizeof(float) * fc_weights_size);
    cudaMalloc(&d_fc_bias_grad, sizeof(float) * 10);

    // Network dimensions for intermediate buffers
    const int N = 128; // Max batch size, adjust if needed
    const int C_in = 1;  // Grayscale input
    const int H_in = 32, W_in = 32;  // Input dimensions
    const int K = 3;     // Kernel size

    // Conv1 dimensions (32 filters)
    const int C_out1 = 32;
    const int H_out1 = H_in - K + 1;  // 30
    const int W_out1 = W_in - K + 1;  // 30

    // Pool1 dimensions
    const int pool_size = 2;
    const int stride = 2;
    const int H_pool1 = (H_out1 - pool_size) / stride + 1;  // 15
    const int W_pool1 = (W_out1 - pool_size) / stride + 1;  // 15

    // Conv2 dimensions (64 filters)
    const int C_out2 = 64;
    const int H_out2 = H_pool1 - K + 1;  // 13
    const int W_out2 = W_pool1 - K + 1;  // 13

    // Pool2 dimensions
    const int H_pool2 = (H_out2 - pool_size) / stride + 1;  // 6
    const int W_pool2 = (W_out2 - pool_size) / stride + 1;  // 6

    // Conv3 dimensions (64 filters)
    const int C_out3 = 64;
    const int H_out3 = H_pool2 - K + 1;  // 4
    const int W_out3 = W_pool2 - K + 1;  // 4

    // Allocate buffers for intermediate results
    cudaMalloc(&d_conv1_output, sizeof(float) * N * C_out1 * H_out1 * W_out1);
    cudaMalloc(&d_pool1_output, sizeof(float) * N * C_out1 * H_pool1 * W_pool1);
    cudaMalloc(&d_conv2_output, sizeof(float) * N * C_out2 * H_out2 * W_out2);
    cudaMalloc(&d_pool2_output, sizeof(float) * N * C_out2 * H_pool2 * W_pool2);
    cudaMalloc(&d_conv3_output, sizeof(float) * N * C_out3 * H_out3 * W_out3);

    // Allocate buffers for intermediate gradients
    cudaMalloc(&d_grad_conv3_output, sizeof(float) * N * C_out3 * H_out3 * W_out3);
    cudaMalloc(&d_grad_pool2_output, sizeof(float) * N * C_out2 * H_pool2 * W_pool2);
    cudaMalloc(&d_grad_conv2_output, sizeof(float) * N * C_out2 * H_out2 * W_out2);
    cudaMalloc(&d_grad_pool1_output, sizeof(float) * N * C_out1 * H_pool1 * W_pool1);
    cudaMalloc(&d_grad_conv1_output, sizeof(float) * N * C_out1 * H_out1 * W_out1);

    cout << "Model architecture:" << endl;
    cout << "  * Input: 1x32x32" << endl;
    cout << "  * Conv1: 32 filters of 3x3 -> 32x30x30" << endl;
    cout << "  * Pool1: 2x2 stride 2 -> 32x15x15" << endl;
    cout << "  * Conv2: 64 filters of 3x3 -> 64x13x13" << endl;
    cout << "  * Pool2: 2x2 stride 2 -> 64x6x6" << endl;
    cout << "  * Conv3: 64 filters of 3x3 -> 64x4x4" << endl;
    cout << "  * FC: 10 outputs (classes)" << endl;
}

// Update destructor to free additional memory
Model::~Model() {
    // Free weights and biases
    cudaFree(d_weights1); cudaFree(d_bias1);
    cudaFree(d_weights2); cudaFree(d_bias2);
    cudaFree(d_weights3); cudaFree(d_bias3);
    cudaFree(d_fc_weights); cudaFree(d_fc_bias);

    // Free velocity buffers
    cudaFree(d_weights1_velocity); cudaFree(d_bias1_velocity);
    cudaFree(d_weights2_velocity); cudaFree(d_bias2_velocity);
    cudaFree(d_weights3_velocity); cudaFree(d_bias3_velocity);
    cudaFree(d_fc_weights_velocity); cudaFree(d_fc_bias_velocity);

    // Free gradient buffers
    cudaFree(d_weights1_grad); cudaFree(d_bias1_grad);
    cudaFree(d_weights2_grad); cudaFree(d_bias2_grad);
    cudaFree(d_weights3_grad); cudaFree(d_bias3_grad);
    cudaFree(d_fc_weights_grad); cudaFree(d_fc_bias_grad);

    // Free intermediate buffers
    cudaFree(d_conv1_output); cudaFree(d_pool1_output);
    cudaFree(d_conv2_output); cudaFree(d_pool2_output);
    cudaFree(d_conv3_output);

    // Free gradient buffers
    cudaFree(d_grad_conv3_output); cudaFree(d_grad_pool2_output);
    cudaFree(d_grad_conv2_output); cudaFree(d_grad_pool1_output);
    cudaFree(d_grad_conv1_output);
}

// Update the forward function to store intermediate outputs
void Model::forward(const float* input, float* output, int N) {
    // Network dimensions
    const int C_in = 1;  // Grayscale input
    const int H_in = 32, W_in = 32;  // Input dimensions
    const int K = 3;     // Kernel size

    // Conv1 dimensions (32 filters)
    const int C_out1 = 32;
    const int H_out1 = H_in - K + 1;  // 30
    const int W_out1 = W_in - K + 1;  // 30

    // Pool1 dimensions
    const int pool_size = 2;
    const int stride = 2;
    const int H_pool1 = (H_out1 - pool_size) / stride + 1;  // 15
    const int W_pool1 = (W_out1 - pool_size) / stride + 1;  // 15

    // Conv2 dimensions (64 filters)
    const int C_out2 = 64;
    const int H_out2 = H_pool1 - K + 1;  // 13
    const int W_out2 = W_pool1 - K + 1;  // 13

    // Pool2 dimensions
    const int H_pool2 = (H_out2 - pool_size) / stride + 1;  // 6
    const int W_pool2 = (W_out2 - pool_size) / stride + 1;  // 6

    // Conv3 dimensions (64 filters)
    const int C_out3 = 64;
    const int H_out3 = H_pool2 - K + 1;  // 4
    const int W_out3 = W_pool2 - K + 1;  // 4

    // FC dimensions
    const int fc_input_size = C_out3 * H_out3 * W_out3;  // 64*4*4 = 1024
    const int fc_output_size = 10;  // 10 classes (CIFAR-10)

    //Layer 1: Convolution
    conv2d_forward(input, d_weights1, d_bias1, d_conv1_output,
        N, C_in, H_in, W_in, C_out1, K);

    //Layer 2: Max Pooling
    maxpool2d_forward(d_conv1_output, d_pool1_output,
        N, C_out1, H_out1, W_out1, pool_size, stride);

    //Layer 3: Convolution
    conv2d_forward(d_pool1_output, d_weights2, d_bias2, d_conv2_output,
        N, C_out1, H_pool1, W_pool1, C_out2, K);

    //Layer 4: Max Pooling
    maxpool2d_forward(d_conv2_output, d_pool2_output,
        N, C_out2, H_out2, W_out2, pool_size, stride);

    //Layer 5: Convolution
    conv2d_forward(d_pool2_output, d_weights3, d_bias3, d_conv3_output,
        N, C_out2, H_pool2, W_pool2, C_out3, K);

    //Layer 6: Flattening, Fully Connected Layer
    dense_forward(d_conv3_output, d_fc_weights, d_fc_bias, output,
        N, fc_input_size, fc_output_size);
}

// Implement backward pass
void Model::backward(const float* input, const float* grad_output, int batch_size) {
    // Network dimensions
    const int C_in = 1;  // Grayscale input
    const int H_in = 32, W_in = 32;  // Input dimensions
    const int K = 3;     // Kernel size

    // Conv1 dimensions (32 filters)
    const int C_out1 = 32;
    const int H_out1 = H_in - K + 1;  // 30
    const int W_out1 = W_in - K + 1;  // 30

    // Pool1 dimensions
    const int pool_size = 2;
    const int stride = 2;
    const int H_pool1 = (H_out1 - pool_size) / stride + 1;  // 15
    const int W_pool1 = (W_out1 - pool_size) / stride + 1;  // 15

    // Conv2 dimensions (64 filters)
    const int C_out2 = 64;
    const int H_out2 = H_pool1 - K + 1;  // 13
    const int W_out2 = W_pool1 - K + 1;  // 13

    // Pool2 dimensions
    const int H_pool2 = (H_out2 - pool_size) / stride + 1;  // 6
    const int W_pool2 = (W_out2 - pool_size) / stride + 1;  // 6

    // Conv3 dimensions (64 filters)
    const int C_out3 = 64;
    const int H_out3 = H_pool2 - K + 1;  // 4
    const int W_out3 = W_pool2 - K + 1;  // 4

    // FC dimensions
    const int fc_input_size = C_out3 * H_out3 * W_out3;  // 64*4*4 = 1024
    const int fc_output_size = 10;  // 10 classes

    // Backward pass for FC layer
    dense_backward(
        d_conv3_output, d_fc_weights, grad_output,
        d_grad_conv3_output, d_fc_weights_grad, d_fc_bias_grad,
        batch_size, fc_input_size, fc_output_size
    );

    // Backward pass for Conv3 layer
    conv2d_backward(
        d_pool2_output, d_weights3, d_grad_conv3_output,
        d_grad_pool2_output, d_weights3_grad, d_bias3_grad,
        batch_size, C_out2, H_pool2, W_pool2, C_out3, K
    );

    // Backward pass for Pool2 layer
    maxpool2d_backward(
        d_conv2_output, d_pool2_output, d_grad_pool2_output, d_grad_conv2_output,
        batch_size, C_out2, H_out2, W_out2, pool_size, stride
    );

    // Backward pass for Conv2 layer
    conv2d_backward(
        d_pool1_output, d_weights2, d_grad_conv2_output,
        d_grad_pool1_output, d_weights2_grad, d_bias2_grad,
        batch_size, C_out1, H_pool1, W_pool1, C_out2, K
    );

    // Backward pass for Pool1 layer
    maxpool2d_backward(
        d_conv1_output, d_pool1_output, d_grad_pool1_output, d_grad_conv1_output,
        batch_size, C_out1, H_out1, W_out1, pool_size, stride
    );

    // Backward pass for Conv1 layer
    conv2d_backward(
        input, d_weights1, d_grad_conv1_output,
        nullptr, d_weights1_grad, d_bias1_grad,  // No need to compute input gradients for the first layer
        batch_size, C_in, H_in, W_in, C_out1, K
    );
}

// Implement parameter update with SGD + momentum
void Model::update(float learning_rate, float momentum) {
    // Network dimensions (for calculating buffer sizes)
    const int weights1_size = 32 * 1 * 3 * 3;
    const int weights2_size = 64 * 32 * 3 * 3;
    const int weights3_size = 64 * 64 * 3 * 3;
    const int fc_weights_size = 10 * (64 * 4 * 4);

    // Update Conv1 parameters
    sgd_update(
        d_weights1, d_weights1_grad, d_weights1_velocity,
        learning_rate, momentum, weights1_size
    );
    sgd_update(
        d_bias1, d_bias1_grad, d_bias1_velocity,
        learning_rate, momentum, 32
    );

    // Update Conv2 parameters
    sgd_update(
        d_weights2, d_weights2_grad, d_weights2_velocity,
        learning_rate, momentum, weights2_size
    );
    sgd_update(
        d_bias2, d_bias2_grad, d_bias2_velocity,
        learning_rate, momentum, 64
    );

    // Update Conv3 parameters
    sgd_update(
        d_weights3, d_weights3_grad, d_weights3_velocity,
        learning_rate, momentum, weights3_size
    );
    sgd_update(
        d_bias3, d_bias3_grad, d_bias3_velocity,
        learning_rate, momentum, 64
    );

    // Update FC parameters
    sgd_update(
        d_fc_weights, d_fc_weights_grad, d_fc_weights_velocity,
        learning_rate, momentum, fc_weights_size
    );
    sgd_update(
        d_fc_bias, d_fc_bias_grad, d_fc_bias_velocity,
        learning_rate, momentum, 10
    );
}
