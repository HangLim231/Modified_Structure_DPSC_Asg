// File: include/loader.h

#pragma once

#include <vector>
#include <string>

#define IMAGE_SIZE 32
#define IMAGE_PIXELS (IMAGE_SIZE * IMAGE_SIZE)
#define NUM_CLASSES 10

struct Image {
    std::vector<float> pixels; // Grayscale normalized
    int label;
};

std::vector<Image> load_dataset(const std::vector<std::string>& batch_files);