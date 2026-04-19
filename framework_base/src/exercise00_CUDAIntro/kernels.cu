#include <cuda_runtime.h>
#include "opg/hostdevice/random.h"
#include "opg/glmwrapper.h"
#include "opg/hostdevice/misc.h"
#include <cstdint>
#include <cmath>

#include "kernels.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ

__global__
void multiply(int n, int* arr, int scalar) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		arr[i] *= scalar;
	}
}

void runMultiply(int n, int* arr, int scalar) {
	multiply<<<256, 256>>>(n, arr, scalar);
	cudaDeviceSynchronize();
}

__global__
void horizontalPass(int n, int width, int height, int channels, uint8_t* img, int* out, int* kernel) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		int x = i % width;
		int y = i / width;

		if (x == 0 || x == width - 1) {
			for (int j = 0; j < channels; j++) {
				out[(y * width + x) * channels + j] = 0;
			}

			continue;
		}

		for (int j = 0; j < channels; j++) {
			int left = img[(y * width + x-1) * channels + j];
			int middle = img[(y * width + x) * channels + j];
			int right = img[(y * width + x+1) * channels + j];

			out[(y * width + x) * channels + j] = kernel[2] * left + kernel[1] * middle + kernel[0] * right;
		}
	}
}

__global__
void verticalPass(int n, int width, int height, int channels, int* img, int* out, int* kernel) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		int x = i % width;
		int y = i / width;

		if (y == 0 || y == height - 1) {
			for (int j = 0; j < channels; j++) {
				out[(y * width + x) * channels + j] = 0;
			}

			continue;
		}

		for (int j = 0; j < channels; j++) {
			int top = img[((y-1) * width + x) * channels + j];
			int middle = img[(y * width + x) * channels + j];
			int bottom = img[((y+1) * width + x) * channels + j];

			out[(y * width + x) * channels + j] = kernel[2] * top + kernel[1] * middle + kernel[0] * bottom;
		}
	}
}

__global__
void computeMagnitude(int n, int channels, uint8_t* out, int* img1, int* img2) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		// accumulate over all color channels
		double val = 0;
		for (int j = 0; j < channels; j++) {
			val += img1[i*channels+j] * img1[i*channels+j] + img2[i*channels+j] * img2[i*channels+j];
		}
		val /= 3.0;
		
		uint8_t final_val = static_cast<uint8_t>(min(std::sqrt(val), 255.0));

		for (int j = 0; j < channels; j++) {
			out[i*channels+j] = final_val;
		}
	}
}

void runHorizontalPass(int n, int width, int height, int channels, uint8_t* img, int* out, int* kernel) {
	horizontalPass<<<256, 256>>>(n, width, height, channels, img, out, kernel);
}

void runVerticalPass(int n, int width, int height, int channels, int* img, int* out, int* kernel) {
	verticalPass<<<256, 256>>>(n, width, height, channels, img, out, kernel);
}

void runComputeMagnitude(int n, int channels, uint8_t* out, int* img1, int* img2) {
	computeMagnitude<<<256, 256>>>(n, channels, out, img1, img2);
}

__global__
void random(int n, float* arr) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	uint32_t seed = sampleTEA32(index, stride);
	PCG32 generator(seed);

	for (int i = index; i < n; i += stride) {
		arr[i] = generator.nextFloat();
	}
}

__global__
void count(int n, float* arr, float threshold, int* result) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		if (arr[i] > threshold) {
			atomicAdd(result, 1);
		}
	}
}

void runRandom(int n, float* arr) {
	random<<<256, 256>>>(n, arr);
}

void runCount(int n, float* arr, float threshold, int* result) {
	count<<<256, 256>>>(n, arr, threshold, result);
}

__global__
void matrixMultiply(int lhsRows, int lhsCols, float* lhs, int rhsRows, int rhsCols, float* rhs, float* out) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < lhsRows * rhsCols; i += stride) {
		int x = i % rhsCols;
		int y = i / rhsCols;

		for (int j = 0; j < rhsRows; j++) {
			out[x + y * rhsCols] += lhs[j + y * lhsCols] * rhs[x + j * rhsCols];
		}
	}
}

void runMatrixMultiply(int lhsRows, int lhsCols, float* lhs, int rhsRows, int rhsCols, float* rhs, float* out) {
	matrixMultiply<<<1, 256>>>(lhsRows, lhsCols, lhs, rhsRows, rhsCols, rhs, out);
}


//
