// File: include/backprop.h

#pragma once

// Backpropagation for dense layer
void dense_backward(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int N, int InputSize, int OutputSize);

// Backpropagation for convolutional layer
void conv2d_backward(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int N, int C_in, int H, int W,
    int C_out, int K);

// Backpropagation for max pooling layer
void maxpool2d_backward(
    const float* d_input,
    const float* d_output,
    const float* d_grad_output,
    float* d_grad_input,
    int N, int C, int H, int W,
    int pool_size, int stride);