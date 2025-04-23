// File: include/model.h

#pragma once

#include <vector>
#include "layers.h"

// Define the neural network model class
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
};
