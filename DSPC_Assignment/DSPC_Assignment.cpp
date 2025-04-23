// File: main.cpp
#include "include/loader.h"
#include "include/evaluate.h"
#include "include/model.h"
#include "include/train_cuda.h"
#include "include/timer.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
using namespace std;

// Function to check CUDA device properties
bool checkCudaDevice() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        cerr << "Error: Failed to get CUDA device count: " << cudaGetErrorString(error) << endl;
        return false;
    }

    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable devices found" << endl;
        return false;
    }

    cout << "Found " << deviceCount << " CUDA-capable device(s)" << endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        cout << "Device " << i << ": " << deviceProp.name << endl;
        cout << "  - Compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "  - Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  - Multiprocessors: " << deviceProp.multiProcessorCount << endl;
        cout << "  - Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
        cout << "  - Max thread dimensions: ("
            << deviceProp.maxThreadsDim[0] << ", "
            << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << ")" << endl;
    }

    // Set device to use
    cudaSetDevice(0);
    cout << "Using device 0 for computation" << endl;

    return true;
}

int main() {
    cout << "\n===== CNN Training: GPU Performance Analysis =====" << endl;

    // Check CUDA device
    if (!checkCudaDevice()) {
        cerr << "CUDA initialization failed. Exiting." << endl;
        return 1;
    }

    // Dataset file paths
    vector<string> batch_files = {
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin"
    };

    // Timing data loading
    Timer loading_timer;
    loading_timer.start();

    cout << "\nLoading dataset from " << batch_files.size() << " files..." << endl;
    vector<Image> dataset = load_dataset(batch_files);

    loading_timer.stop();

    if (dataset.empty()) {
        cerr << "Error: Dataset is empty. Check file paths and permissions." << endl;
        return 1;
    }

    cout << "Dataset loaded successfully! " << endl;
    cout << "  - Total images: " << dataset.size() << endl;
    cout << "  - Image dimensions: " << IMAGE_SIZE << "x" << IMAGE_SIZE << " pixels" << endl;
    cout << "  - Loading time: " << fixed << setprecision(2) << loading_timer.elapsedMilliseconds() << " ms" << endl;

    // Distribution of classes in the dataset
    vector<int> class_distribution(NUM_CLASSES, 0);
    for (const auto& img : dataset) {
        if (img.label >= 0 && img.label < NUM_CLASSES) {
            class_distribution[img.label]++;
        }
    }

    cout << "\nClass distribution:" << endl;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        cout << "  - Class " << i << ": " << class_distribution[i] << " images" << endl;
    }

    // Timing GPU training
    cout << "\nStarting CUDA training..." << endl;
    Timer training_timer;
    training_timer.start();

    // Train the model using CUDA
    train_cuda(dataset);

    training_timer.stop();
    cout << "\nCUDA training completed!" << endl;
    cout << "  - Total training time: " << fixed << setprecision(2)
        << training_timer.elapsedMilliseconds() << " ms" << endl;

    // Check for any CUDA errors after training
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "CUDA error after training: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    // Clean up CUDA
    cudaDeviceReset();

    cout << "\n===== Training Complete =====" << endl;
    return 0;
}