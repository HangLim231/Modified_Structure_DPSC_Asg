// File: include/layers.h

#pragma once


namespace layers {

    void conv2d_forward(
        const float* d_input,
        const float* d_weights,
        const float* d_bias,
        float* d_output,
        int N, int C_in, int H, int W,
        int C_out, int K
    );

    void maxpool2d_forward(
        const float* d_input,
        float* d_output,
        int N, int C, int H, int W,
        int pool_size, int stride
    );

    void dense_forward(
        const float* d_input,
        const float* d_weights,
        const float* d_bias,
        float* d_output,
        int N, int InputSize, int OutputSize
    );

} // namespace layers