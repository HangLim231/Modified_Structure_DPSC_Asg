// File: include/model.h

#pragma once

#include <vector>
#include "layers.h"

class Model {
public:
    Model();
    ~Model();

    float* d_weights1;
    float* d_bias1;
    float* d_weights2;
    float* d_bias2;
    float* d_weights3;
    float* d_bias3;
    float* d_fc_weights;
    float* d_fc_bias;

    void initialize();
    void forward(const float* input, float* output, int N);
};
