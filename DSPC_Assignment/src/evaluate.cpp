// File: src/evaluate.cpp

#include "../include/evaluate.h"
using namespace std;

// Function to find the index of the maximum value in an array
// This function is used to determine the predicted class from the model's output logits.
int argmax(const float* logits, int num_classes) {
    int max_index = 0;
    float max_value = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_value) {
            max_value = logits[i];
            max_index = i;
        }
    }
    return max_index;
}

// Function to calculate accuracy
float calculate_accuracy(const vector<int>& predictions, const vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size();
}
