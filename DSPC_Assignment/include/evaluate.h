// File: include/evaluate.h

#pragma once

#include <vector>
using namespace std;

float calculate_accuracy(const vector<int>& predictions, const vector<int>& labels);
int argmax(const float* logits, int num_classes);