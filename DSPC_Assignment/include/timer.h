// File: include/timer.h

#pragma once

#include <chrono>

// Timer class to measure elapsed time
class Timer {
public:
    Timer();
    void start();   // Start the timer
    void stop();    // Stop the timer
    double elapsedMilliseconds() const; // Get elapsed time in milliseconds
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};
