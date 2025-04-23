// File: src/model.cpp

#include "../include/model.h"
#include "../include/timer.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace layers;

Model::Model() : d_weights1(nullptr), d_bias1(nullptr),
d_weights2(nullptr), d_bias2(nullptr),
d_weights3(nullptr), d_bias3(nullptr),
d_fc_weights(nullptr), d_fc_bias(nullptr) {}

Model::~Model() {
    cudaFree(d_weights1); cudaFree(d_bias1);
    cudaFree(d_weights2); cudaFree(d_bias2);
    cudaFree(d_weights3); cudaFree(d_bias3);
    cudaFree(d_fc_weights); cudaFree(d_fc_bias);
}

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

    cout << "Model architecture:" << endl;
    cout << "  * Input: 1x32x32" << endl;
    cout << "  * Conv1: 32 filters of 3x3 -> 32x30x30" << endl;
    cout << "  * Pool1: 2x2 stride 2 -> 32x15x15" << endl;
    cout << "  * Conv2: 64 filters of 3x3 -> 64x13x13" << endl;
    cout << "  * Pool2: 2x2 stride 2 -> 64x6x6" << endl;
    cout << "  * Conv3: 64 filters of 3x3 -> 64x4x4" << endl;
    cout << "  * FC: 10 outputs (classes)" << endl;
}

void Model::forward(const float* input, float* output, int N) {
    // Full CNN forward pass implementation
    Timer layer_timer;

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

    // Allocate memory for intermediate results
    float* d_conv1, * d_pool1, * d_conv2, * d_pool2, * d_conv3, * d_flattened;
    cudaMalloc(&d_conv1, sizeof(float) * N * C_out1 * H_out1 * W_out1);
    cudaMalloc(&d_pool1, sizeof(float) * N * C_out1 * H_pool1 * W_pool1);
    cudaMalloc(&d_conv2, sizeof(float) * N * C_out2 * H_out2 * W_out2);
    cudaMalloc(&d_pool2, sizeof(float) * N * C_out2 * H_pool2 * W_pool2);
    cudaMalloc(&d_conv3, sizeof(float) * N * C_out3 * H_out3 * W_out3);
    cudaMalloc(&d_flattened, sizeof(float) * N * fc_input_size);

    //Layer 1: Convolution
    conv2d_forward(input, d_weights1, d_bias1, d_conv1,
        N, C_in, H_in, W_in, C_out1, K);

    //Layer 2: Max Pooling
    maxpool2d_forward(d_conv1, d_pool1, N, C_out1, H_out1, W_out1, pool_size, stride);

    //Layer 3: Convolution
    conv2d_forward(d_pool1, d_weights2, d_bias2, d_conv2,
        N, C_out1, H_pool1, W_pool1, C_out2, K);

    //Layer 4: Max Pooling
    maxpool2d_forward(d_conv2, d_pool2, N, C_out2, H_out2, W_out2, pool_size, stride);

    //Layer 5: Convolution
    conv2d_forward(d_pool2, d_weights3, d_bias3, d_conv3,
        N, C_out2, H_pool2, W_pool2, C_out3, K);

    //Layer 6: Flattening, Fully Connected Layer
    float* d_flattened_view = d_conv3;
    dense_forward(d_flattened_view, d_fc_weights, d_fc_bias, output,
        N, fc_input_size, fc_output_size);

    // Free intermediate buffers
    cudaFree(d_conv1);
    cudaFree(d_pool1);
    cudaFree(d_conv2);
    cudaFree(d_pool2);
    cudaFree(d_conv3);
}