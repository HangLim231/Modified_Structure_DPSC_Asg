// File: include/evaluate.h

#pragma once

#include <vector>
using namespace std;

// Function to calculate accuracy
float calculate_accuracy(const vector<int>& predictions, const vector<int>& labels);
// Function to find the index of the maximum value in an array
int argmax(const float* logits, int num_classes);