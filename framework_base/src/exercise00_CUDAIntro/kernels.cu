#include <cuda_runtime.h>
#include "opg/hostdevice/random.h"
#include "opg/glmwrapper.h"
#include "opg/hostdevice/misc.h"
#include <cstdint>

#include "kernels.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ
__global__
void multiplyKernel(int *d_dataArray, int constant, int size) {
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int stride = blockDim.x * gridDim.x;
     for (int i = index; i < size; i += stride) {
         d_dataArray[i] *= constant;
     }

}
void multiplyByConstant(int *d_dataArray, int constant, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(d_dataArray, constant, size);
}
__global__
void horizontalSobel(float* input, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width -1 && y < height){
        for(int c = 0; c < channels; ++c) {
            int center_idx = (y *width +x) *channels + c;
            int left_idx = (y *width + (x-1)) *channels + c;
            int right_idx = (y *width + (x+1)) *channels + c;
            output[center_idx] = input[right_idx] - input[left_idx];



        }
}
}
__global__
void verticalSobel(float* input, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y >= 1 && y < height - 1){
        for(int c = 0; c < channels; ++c) {
            int center_idx = (y *width +x) *channels + c;
            int top_idx = ((y-1) *width + x) *channels + c;
            int bottom_idx = ((y+1) *width + x) *channels + c;
            output[center_idx] = input[bottom_idx] - input[top_idx];




        }
}
}
__global__ 
void verticalSobelSmoothing(float* input, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y >= 1 && y < height - 1){
        for(int c = 0; c < channels; ++c) {
            int center_idx = (y *width +x) *channels + c;
            int top_idx = ((y-1) *width + x) *channels + c;
            int bottom_idx = ((y+1) *width + x) *channels + c;
            output[center_idx] = input[top_idx] + (2.0f * input[center_idx]) + input[bottom_idx];
        }
}
}
__global__ 
void horizontalSobelSmoothing(float* input, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width -1 && y < height){
        for(int c = 0; c < channels; ++c) {
            int center_idx = (y *width +x) *channels + c;
            int left_idx = (y *width + (x-1)) *channels + c;
            int right_idx = (y *width + (x+1)) *channels + c;
            output[center_idx] = input[left_idx] + (2.0f * input[center_idx]) + input[right_idx];
        }
}
}
__global__
void computeMagintude(float* horizontal, float* vertical, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height){
        for(int c = 0; c < channels; ++c) {
            int center_idx = (y *width +x) *channels + c;
            float h = horizontal[center_idx];
            float v = vertical[center_idx];
            output[center_idx] = sqrt(h * h + v * v);
        }
}
}
void applySobelFilter(float *input_image, float *output_image, int width, int height, int channels) {
    int imageSize = width * height * channels * sizeof(float);
    float *d_temp, *d_horizontal, *d_vertical;
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_horizontal, imageSize);
    cudaMalloc(&d_vertical, imageSize);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    horizontalSobel<<<gridSize, blockSize>>>(input_image, d_temp, width, height, channels);
    cudaDeviceSynchronize();
    
    verticalSobelSmoothing<<<gridSize, blockSize>>>(d_temp, d_horizontal, width, height, channels);
    cudaDeviceSynchronize();

    verticalSobel<<<gridSize, blockSize>>>(input_image, d_temp, width, height, channels);
    cudaDeviceSynchronize();
    
    horizontalSobelSmoothing<<<gridSize, blockSize>>>(d_temp, d_vertical, width, height, channels);
    cudaDeviceSynchronize();

    computeMagintude<<<gridSize, blockSize>>>(d_horizontal, d_vertical, output_image, width, height, channels);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
    cudaFree(d_horizontal);
    cudaFree(d_vertical);
}