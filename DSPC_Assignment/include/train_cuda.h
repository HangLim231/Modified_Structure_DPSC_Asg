//File name: include/train_cuda.h
#pragma once
#include "loader.h"
#include <vector>

// Function to call the train model
void train_cuda(const std::vector<Image>& dataset);
