// File: src/evaluate.cpp

#include "../include/evaluate.h"
using namespace std;

float calculate_accuracy(const vector<int>& predictions, const vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size();
}
