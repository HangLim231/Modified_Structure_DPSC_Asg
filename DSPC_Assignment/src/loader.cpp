// File: src/loader.cpp

#include "../include/loader.h"
#include <fstream>
#include <iostream>
using namespace std;

// Function to load the CIFAR-10 dataset
vector<Image> load_dataset(const vector<string>& batch_files) {
    vector<Image> dataset;
    const int record_size = 1 + 3072;
    unsigned char buffer[record_size];

    for (const auto& file_path : batch_files) {
        ifstream file(file_path, ios::binary);
        if (!file) {
            cerr << "Error: Failed to open " << file_path << endl;
            continue;
        }

        while (file.read(reinterpret_cast<char*>(buffer), record_size)) {
            Image img;
            img.label = buffer[0];
            img.pixels.resize(IMAGE_PIXELS);

			// Convert RGB to grayscale 
            for (int i = 0; i < IMAGE_PIXELS; ++i) {
                float r = static_cast<float>(buffer[1 + i]) / 255.0f;
                float g = static_cast<float>(buffer[1 + i + 1024]) / 255.0f;
                float b = static_cast<float>(buffer[1 + i + 2048]) / 255.0f;
                img.pixels[i] = (r + g + b) / 3.0f;
            }

            dataset.push_back(img);
        }
    }

    return dataset;
}