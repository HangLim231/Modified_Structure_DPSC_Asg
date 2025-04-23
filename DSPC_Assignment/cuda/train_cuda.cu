// File: cuda/train_cuda.cu

#include "../include/model.h"
#include "../include/loader.h"
#include "../include/evaluate.h"
#include "../include/train_cuda.h"
#include "../include/timer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include "../include/loss.h"
using namespace std;

// Utility function to check CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
             << cudaGetErrorString(error) << endl; \
        return; \
    } \
} while(0)

// Function to create batches from dataset
vector<vector<int>> createBatches(int dataset_size, int batch_size) {
    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) {
        indices[i] = i;
    }

    // Shuffle indices for random batches
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    vector<vector<int>> batches;
    for (int i = 0; i < dataset_size; i += batch_size) {
        vector<int> batch;
        for (int j = i; j < min(i + batch_size, dataset_size); j++) {
            batch.push_back(indices[j]);
        }
        batches.push_back(batch);
    }

    return batches;
}

// Function to prepare a batch
void prepareBatch(const vector<Image>& dataset, const vector<int>& batch_indices,
    float* batch_data, int* batch_labels) {
    int batch_size = batch_indices.size();

    for (int i = 0; i < batch_size; i++) {
        int idx = batch_indices[i];
        // Copy image pixels
        copy(dataset[idx].pixels.begin(), dataset[idx].pixels.end(),
            batch_data + i * IMAGE_PIXELS);
        // Copy label
        batch_labels[i] = dataset[idx].label;
    }
}

// Function to evaluate model on a batch
float evaluateBatch(float* d_input, int* d_labels, Model& model, int batch_size) {
    float* d_output;
    cudaMalloc(&d_output, sizeof(float) * batch_size * NUM_CLASSES);

    // Forward pass
    model.forward(d_input, d_output, batch_size);

    // Copy output back to host
    float* h_output = new float[batch_size * NUM_CLASSES];
    int* h_labels = new int[batch_size];

    cudaMemcpy(h_output, d_output, sizeof(float) * batch_size * NUM_CLASSES,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels, d_labels, sizeof(int) * batch_size,
        cudaMemcpyDeviceToHost);

    // Compute accuracy
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        int pred_class = argmax(h_output + i * NUM_CLASSES, NUM_CLASSES);
        if (pred_class == h_labels[i]) {
            correct++;
        }
    }

    float accuracy = static_cast<float>(correct) / batch_size;

    // Clean up
    delete[] h_output;
    delete[] h_labels;
    cudaFree(d_output);

    return accuracy;
}


// Function to train model on a batch
void trainBatch(float* d_input, int* d_labels, Model& model, int batch_size, float learning_rate, float momentum) {
    // Allocate memory for network output
    float* d_output;
    cudaMalloc(&d_output, sizeof(float) * batch_size * NUM_CLASSES);

    // Forward pass
    model.forward(d_input, d_output, batch_size);

    // Allocate memory for loss values and gradients
    float* d_losses;
    float* d_gradients;
    cudaMalloc(&d_losses, sizeof(float) * batch_size);
    cudaMalloc(&d_gradients, sizeof(float) * batch_size * NUM_CLASSES);

    // Compute loss and its gradients
    float loss = compute_loss(d_output, d_labels, d_losses, batch_size, NUM_CLASSES);
    compute_loss_gradients(d_output, d_labels, d_gradients, batch_size, NUM_CLASSES);

    // Backward pass through the network
    model.backward(d_input, d_gradients, batch_size);

    // Update model parameters using SGD with momentum
    model.update(learning_rate, momentum);

    // Clean up
    cudaFree(d_output);
    cudaFree(d_losses);
    cudaFree(d_gradients);
}

// Main function to train the model using CUDA
void train_cuda(const vector<Image>& dataset) {
    // Training hyperparameters
    const int epochs = 10;
    const int batch_size = 128;
    const int validation_size = 1000;  // Number of samples for validation

    // Constants
    int N = static_cast<int>(dataset.size());
    int train_size = N - validation_size;

    cout << "\n===== CUDA CNN Training =====" << endl;
    cout << "Dataset size: " << N << " images" << endl;
    cout << "Training samples: " << train_size << endl;
    cout << "Validation samples: " << validation_size << endl;
    cout << "Batch size: " << batch_size << endl;
    cout << "Epochs: " << epochs << endl;

    // Split into training and validation
    vector<Image> train_data(dataset.begin(), dataset.begin() + train_size);
    vector<Image> val_data(dataset.begin() + train_size, dataset.end());

    // Allocate host memory for batch
    float* h_batch_data = new float[batch_size * IMAGE_PIXELS];
    int* h_batch_labels = new int[batch_size];

    // Allocate device memory for batch
    float* d_batch_data;
    int* d_batch_labels;
    CUDA_CHECK(cudaMalloc(&d_batch_data, sizeof(float) * batch_size * IMAGE_PIXELS));
    CUDA_CHECK(cudaMalloc(&d_batch_labels, sizeof(int) * batch_size));

    // Initialize model
    cout << "\nInitializing model..." << endl;
    Model model;
    model.initialize();

    // For validation data
    float* d_val_data;
    int* d_val_labels;
    CUDA_CHECK(cudaMalloc(&d_val_data, sizeof(float) * validation_size * IMAGE_PIXELS));
    CUDA_CHECK(cudaMalloc(&d_val_labels, sizeof(int) * validation_size));

    // Prepare validation data
    float* h_val_data = new float[validation_size * IMAGE_PIXELS];
    int* h_val_labels = new int[validation_size];

    for (int i = 0; i < validation_size; i++) {
        copy(val_data[i].pixels.begin(), val_data[i].pixels.end(),
            h_val_data + i * IMAGE_PIXELS);
        h_val_labels[i] = val_data[i].label;
    }

    CUDA_CHECK(cudaMemcpy(d_val_data, h_val_data, sizeof(float) * validation_size * IMAGE_PIXELS,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val_labels, h_val_labels, sizeof(int) * validation_size,
        cudaMemcpyHostToDevice));

    Timer total_timer;
    total_timer.start();

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "\nEpoch " << epoch + 1 << "/" << epochs << endl;

        vector<vector<int>> batches = createBatches(train_size, batch_size);
        float epoch_accuracy = 0.0f;
        int total_batches = batches.size();

        // Process each batch
        for (size_t batch_idx = 0; batch_idx < batches.size(); batch_idx++) {
            const auto& batch_indices = batches[batch_idx];
            int current_batch_size = batch_indices.size();

            // Prepare batch
            prepareBatch(train_data, batch_indices, h_batch_data, h_batch_labels);
            CUDA_CHECK(cudaMemcpy(d_batch_data, h_batch_data,
                sizeof(float) * current_batch_size * IMAGE_PIXELS,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_batch_labels, h_batch_labels,
                sizeof(int) * current_batch_size,
                cudaMemcpyHostToDevice));

            // Train on the batch
            float learning_rate = 0.01f;  // You may want to adjust this or implement learning rate decay
            float momentum = 0.9f;        // Typical momentum value
            trainBatch(d_batch_data, d_batch_labels, model, current_batch_size, learning_rate, momentum);

            // Evaluate the batch for progress tracking
            float batch_accuracy = evaluateBatch(d_batch_data, d_batch_labels, model, current_batch_size);
            epoch_accuracy += batch_accuracy;

            // Print progress every 10%
            if (batch_idx % (total_batches / 10) == 0) {
                cout << "  Progress: " << (batch_idx * 100) / total_batches << "%, "
                    << "Batch Accuracy: " << fixed << setprecision(2) << batch_accuracy * 100 << "%" << endl;
            }
        }

        epoch_accuracy /= total_batches;
        float val_accuracy = evaluateBatch(d_val_data, d_val_labels, model, validation_size);

        cout << "Epoch " << epoch + 1 << " Summary:" << endl;
        cout << "  Training Accuracy: " << fixed << setprecision(2) << epoch_accuracy * 100 << "%" << endl;
        cout << "  Validation Accuracy: " << fixed << setprecision(2) << val_accuracy * 100 << "%" << endl;
    }

    total_timer.stop();
    cout << "\n===== Training Summary =====" << endl;
    cout << "Total training time: " << fixed << setprecision(2)
        << total_timer.elapsedSeconds() << " s" << endl;


    // Clean up
    delete[] h_batch_data;
    delete[] h_batch_labels;
    delete[] h_val_data;
    delete[] h_val_labels;
        
    cudaFree(d_batch_data);
    cudaFree(d_batch_labels);
    cudaFree(d_val_data);
    cudaFree(d_val_labels);
}