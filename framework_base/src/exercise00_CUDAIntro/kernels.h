#pragma once

// Declare your kernel functions here:
void multiplyByConstant(int *d_dataArray, int constant, int size);
void applySobelFilter(float *input_image, float *output_image, int width, int height, int channels);
