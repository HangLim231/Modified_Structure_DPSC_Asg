// File: train/train_cuda.cpp

#include "../include/model.h"
#include "../include/loader.h"
#include "../include/evaluate.h"
#include "../include/train_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

//Main function to train the model using CUDA
void train_cuda(const vector<Image>& dataset) {
    // Constants
    int N = static_cast<int>(dataset.size());   
    size_t input_size = N * IMAGE_PIXELS * sizeof(float); // Number of images * number of pixels per image
    size_t label_size = N * sizeof(int);

    float* d_input;
    int* d_labels;

    // Allocate device memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_labels, label_size);

    // Flatten the dataset
    vector<float> input_flat(N * IMAGE_PIXELS);
    vector<int> label_flat(N);

    // Copy data to the flattened vectors
    for (int i = 0; i < N; ++i) {
        copy(dataset[i].pixels.begin(), dataset[i].pixels.end(), input_flat.begin() + i * IMAGE_PIXELS);
        label_flat[i] = dataset[i].label;
    }

    // Copy data from host to device
    cudaMemcpy(d_input, input_flat.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, label_flat.data(), label_size, cudaMemcpyHostToDevice);

    // Initialize the model
    Model model;
    model.initialize();

    // Allocate output memory
    float* d_output;
    cudaMalloc(&d_output, sizeof(float) * N * NUM_CLASSES);

    model.forward(d_input, d_output, N);

    
    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_output);
}