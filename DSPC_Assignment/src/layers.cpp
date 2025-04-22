//File name: src/layers.cpp

#include "../include/layers.h"
using namespace std;

// Forward declarations of CUDA functions
void conv2d_forward(const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output, int N, int C_in, int H, int W, int C_out, int K);

void maxpool2d_forward(const float* d_input, float* d_output,
    int N, int C, int H, int W, int pool_size, int stride);

void dense_forward(const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output, int N, int InputSize, int OutputSize);

namespace layers {

    void conv2d_forward(const float* d_input, const float* d_weights, const float* d_bias,
        float* d_output, int N, int C_in, int H, int W, int C_out, int K) {
        ::conv2d_forward(d_input, d_weights, d_bias, d_output, N, C_in, H, W, C_out, K);
    }

    void maxpool2d_forward(const float* d_input, float* d_output,
        int N, int C, int H, int W, int pool_size, int stride) {
        ::maxpool2d_forward(d_input, d_output, N, C, H, W, pool_size, stride);
    }

    void dense_forward(const float* d_input, const float* d_weights, const float* d_bias,
        float* d_output, int N, int InputSize, int OutputSize) {
        ::dense_forward(d_input, d_weights, d_bias, d_output, N, InputSize, OutputSize);
    }

}