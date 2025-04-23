// File: include/optimizer.h

#pragma once

// SGD optimizer update with momentum
void sgd_update(
    float* d_params,
    const float* d_gradients,
    float* d_velocities,
    float learning_rate,
    float momentum,
    int size);