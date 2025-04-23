// File: include/loss.h

#pragma once

// Calculate cross entropy loss
float compute_loss(
    const float* d_predictions,
    const int* d_targets,
    float* d_losses,
    int N, int num_classes);

// Compute gradients for cross entropy loss
void compute_loss_gradients(
    const float* d_predictions,
    const int* d_targets,
    float* d_gradients,
    int N, int num_classes);