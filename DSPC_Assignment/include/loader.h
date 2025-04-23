// File: include/loader.h

#pragma once

#include <vector>
#include <string>

#define IMAGE_SIZE 32
#define IMAGE_PIXELS (IMAGE_SIZE * IMAGE_SIZE)
#define NUM_CLASSES 10

// CIFAR-10 dataset structure
struct Image {
    std::vector<float> pixels; // Stored RGB channels
    int label;
};

std::vector<Image> load_dataset(const std::vector<std::string>& batch_files);