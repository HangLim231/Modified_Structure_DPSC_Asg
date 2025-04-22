// File: main.cpp
#include "include/loader.h"
#include "include/evaluate.h"
#include "include/model.h"
#include <iostream>
#include <vector>
using namespace std;

// Declare the function prototype instead of including the .cu file
void train_cuda(const vector<Image>& dataset);

int main() {
    vector<string> batch_files = {
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
        "data/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin"
    };

    vector<Image> dataset = load_dataset(batch_files);
    if (dataset.empty()) {
        cout << "Dataset is empty." << endl;
        return 1;
    }

    train_cuda(dataset);

    cout << "Training complete." << endl;
    return 0;
}
