// File: src/timer.cpp

#include "../include/timer.h"

// Performance measurement utilities
Timer::Timer() {}

// Start the timer
void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

// Stop the timer
void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
}

// Get elapsed time in milliseconds
double Timer::elapsedSeconds() const {
    return std::chrono::duration<double>(end_time - start_time).count();
}
