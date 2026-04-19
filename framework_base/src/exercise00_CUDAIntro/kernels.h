#pragma once

#include <cstdint>

// Declare your kernel functions here:

void runMultiply(int n, int* arr, int scalar);

void runHorizontalPass(int n, int width, int height, int channels, uint8_t* img, int* out, int* kernel);
void runVerticalPass(int n, int width, int height, int channels, int* img, int* out, int* kernel);
void runComputeMagnitude(int n, int channels, uint8_t* out, int* img1, int* img2);

void runRandom(int n, float* arr);
void runCount(int n, float* arr, float threshold, int* result);

void runMatrixMultiply(int lhsRows, int lhsCols, float* lhs, int rhsRows, int rhsCols, float* rhs, float* out);
//
