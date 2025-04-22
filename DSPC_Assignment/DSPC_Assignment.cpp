// File: main.cpp
#include "include/loader.h"
#include "include/evaluate.h"
#include "include/model.h"
#include <iostream>
#include <vector>
#include "cuda/train_cuda.cu"
using namespace std;

int main() {
    vector<string> batch_files = {
        "data/cifar-10-binary/data_batch_1.bin"
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