// File: train/train_cuda.cpp

#include "../include/model.h"
#include "../include/loader.h"
#include "../include/evaluate.h"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

void train_cuda(const vector<Image>& dataset) {
    int N = static_cast<int>(dataset.size());
    size_t input_size = N * IMAGE_PIXELS * sizeof(float);
    size_t label_size = N * sizeof(int);

    float* d_input;
    int* d_labels;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_labels, label_size);

    vector<float> input_flat(N * IMAGE_PIXELS);
    vector<int> label_flat(N);
    for (int i = 0; i < N; ++i) {
        copy(dataset[i].pixels.begin(), dataset[i].pixels.end(), input_flat.begin() + i * IMAGE_PIXELS);
        label_flat[i] = dataset[i].label;
    }

    cudaMemcpy(d_input, input_flat.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, label_flat.data(), label_size, cudaMemcpyHostToDevice);

    Model model;
    model.initialize();

    float* d_output;
    cudaMalloc(&d_output, sizeof(float) * N * NUM_CLASSES);

    model.forward(d_input, d_output, N);

    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_output);
}