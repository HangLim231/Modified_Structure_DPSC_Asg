// File: include/model.h
#pragma once
#include <vector>
#include "layers.h"

// Model class for the neural network
class Model {
public:
    Model();
    ~Model();

    // Weights and biases for the different layers
    float* d_weights1;
    float* d_bias1;
    float* d_weights2;
    float* d_bias2;
    float* d_weights3;
    float* d_bias3;
    float* d_fc_weights;
    float* d_fc_bias;

    // Setup the model parameters
    void initialize();

    // Perform forward pass
    void forward(const float* input, float* output, int N);

    // Perform backward pass
    void backward(const float* input, const float* grad_output, int batch_size);

    // Update parameters using SGD with momentum
    void update(float learning_rate, float momentum);

private:
    // Velocity buffers for SGD with momentum
    float* d_weights1_velocity;
    float* d_bias1_velocity;
    float* d_weights2_velocity;
    float* d_bias2_velocity;
    float* d_weights3_velocity;
    float* d_bias3_velocity;
    float* d_fc_weights_velocity;
    float* d_fc_bias_velocity;

    // Gradient buffers
    float* d_weights1_grad;
    float* d_bias1_grad;
    float* d_weights2_grad;
    float* d_bias2_grad;
    float* d_weights3_grad;
    float* d_bias3_grad;
    float* d_fc_weights_grad;
    float* d_fc_bias_grad;

    // Intermediate results needed for backpropagation
    float* d_conv1_output;
    float* d_pool1_output;
    float* d_conv2_output;
    float* d_pool2_output;
    float* d_conv3_output;

    // Intermediate gradients
    float* d_grad_conv3_output;
    float* d_grad_pool2_output;
    float* d_grad_conv2_output;
    float* d_grad_pool1_output;
    float* d_grad_conv1_output;
};