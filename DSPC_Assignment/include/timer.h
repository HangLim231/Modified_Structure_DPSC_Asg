// File: include/timer.h

#pragma once

#include <chrono>

class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsedMilliseconds() const;
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};
