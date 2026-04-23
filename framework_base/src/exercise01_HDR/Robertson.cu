#include "Robertson.h"

#include "opg/hostdevice/misc.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#define EPSILON 1e-16 

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ

template <class Vec3T>
__global__ void splitChannelsKernel(Vec3T* pixels, typename Vec3T::value_type* red, typename Vec3T::value_type* green, typename Vec3T::value_type* blue, int number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    const uint32_t img_index = gid;
    red[gid] = pixels[img_index].x;
    green[gid] = pixels[img_index].y;
    blue[gid] = pixels[img_index].z;
}

template <class Vec3T>
__global__ void mergeChannelsKernel(Vec3T* pixels, typename Vec3T::value_type* red, typename Vec3T::value_type* green, typename Vec3T::value_type* blue, int number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    const uint32_t img_index = gid;
    pixels[img_index].x = red[gid];
    pixels[img_index].y = green[gid];
    pixels[img_index].z = blue[gid];
}

__global__ void calcMaskKernel(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image)
{

    // values per image is usually width*height
    // number values is width*height*num_images

    // This function filters out Pixels whose Values are overall under or over-exposed across all images. 
    // Thus this pixel would skew with the overall result by raising or lowering the mean in the coming steps

    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x; //grid index
    if (gid >= number_values) // Stop when we have looked at all pixels
    {
        return;
    }

    uint32_t number_imgs = number_values / values_per_image; // Is devision on Cumpute Units more expensive than multiplication?
    float mean = 0.0f;
    for (size_t i = 0; i < number_imgs; i++)
    {
        mean += float(values[gid + i * values_per_image]) / float(number_imgs);
    }

    // Mask out under- *and* overexposed pixels.
    underexposed_mask[gid] = (mean < 5.0f) || (mean > 250.0f);
}

__global__ void countValuesKernel(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image)
{

    // values per image is usually width*height
    // number values is width*height*num_images

    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values) // Stop when we have looked at all pixels
    {
        return;
    }
    
    size_t j = gid % values_per_image; // Points to the current element's position within a single image
    if (underexposed_mask[j]) // Skip pixels that were under or overexposed overall anyway
    {
        return;
    }

    atomicAdd(counters + values[gid], 1); // 
}

__global__ void calcWeightsKernel(uint8_t* values, float* weights, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    float nom = float(values[gid]) - 127.5f;
    float denom = 127.5f * 127.5f;
    float w = fmaxf(expf(-4.0f * nom * nom / denom) - expf(-4.0f), 0.0f); // See (5) in the Paper
    weights[gid] = w;
}

__global__ void normInvCrfKernel(float* I, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    float ref = I[(number_values - 1) / 2];
    I[gid] /= ref;
}

__global__ void initInvCrfKernel(float* I, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    I[gid] = float(gid);
}

// TODO: put your CUDA kernels and the host functions which launch the kernels here
//

__global__
void calcLightValsDivKernel(float* x_hat, float* x_num, float* x_denom, uint32_t num_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_values)
    {
        return;
    }

    if (x_denom[gid] < EPSILON){
        // printf("WARNING: Division by near-zero denominator detected at index %u (Value: %.6f). Setting x_denom to e.\n", gid, x_denom[gid]);
        x_denom[gid] = EPSILON;   
    }
    // x_hat[gid] = x_num[gid] / x_denom[gid];
    float result = x_num[gid] / x_denom[gid];
    
    x_hat[gid] = result;
}

__global__
void calcLightValsNumeratorKernel(float* x_num, uint8_t* pixels, float* weights, float* exposures, float* I, uint32_t num_imgs, uint32_t num_values, uint32_t iteration)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= (num_values * num_imgs))
    {
        return;
    }

    const uint32_t i = gid / num_values; // Logically this should give me the right t index for the exposures array
    const uint32_t j = gid % num_values; // counter for which pixel we are currently on
    const uint8_t yij = pixels[gid];

    float value = weights[gid] * exposures[i] * I[yij];

    atomicAdd(x_num + j, value); // TODO: why no atomic add for floats
    
}

__global__
void calcLightValsDenominatorKernel(float* x_denom, uint8_t* pixels, float* weights, float* exposures, uint32_t num_imgs, uint32_t num_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_values * num_imgs)
    {
        return;
    }

    const uint32_t i = gid / num_values; // Logically this should give me the right t index for the exposures array
    const uint32_t j = gid % num_values; // counter for which pixel we are currently on

    float value = weights[gid] * pow(exposures[i],2.0f);

    atomicAdd(x_denom + j, value); // Need to give a refernce for this to work ???
}

__global__
void calcIEstimSumKernel(float* I_hat_unscaled, float* exposures, float* x, uint8_t* pixels, bool* underexposed_mask, uint32_t num_imgs, uint32_t num_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= (num_values * num_imgs))
    {
        return;
    }

    const uint32_t i = gid / num_values; // Logically this should give me the right t index for the exposures array
    const uint32_t j = gid % num_values; // counter for which pixel we are currently on

    if (underexposed_mask[j])
    {
        return;
    }

    const uint32_t yij = pixels[gid];

    float value = exposures[i] * x[j];

    atomicAdd(I_hat_unscaled + yij, value);
}

__global__
void calcIEstimScaleKernel(float* I_hat, uint32_t* card_Em, uint32_t vals){
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= (vals))
    {
        return;
    }

    if (card_Em[gid] == 0)
    {
        I_hat[gid] = 0.0f;
        return;
    }

    float scale = 1.0f / float(card_Em[gid]);
    I_hat[gid] *= scale;
}

/////////////////////
/////////////////////
/////////////////////

void splitChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    splitChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);

    cudaDeviceSynchronize();
}

void splitChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    splitChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);

    cudaDeviceSynchronize();
}

void mergeChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    mergeChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);

    cudaDeviceSynchronize();
}

void mergeChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_pixels, block_size); // Spawn enough blocks
    mergeChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);

    cudaDeviceSynchronize();
}

void calcMask(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(values_per_image, block_size); // Spawn enough blocks
    calcMaskKernel<<<block_count, block_size>>>(values, underexposed_mask, number_values, values_per_image);

    cudaDeviceSynchronize();
}

void countValues(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
    countValuesKernel<<<block_count, block_size>>>(values, underexposed_mask, counters, number_values, values_per_image);

    cudaDeviceSynchronize();
}

void calcWeights(uint8_t* values, float* weights, uint32_t number_values)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
    calcWeightsKernel<<<block_count, block_size>>>(values, weights, number_values);

    cudaDeviceSynchronize();
}

void initInvCrf(float* I, uint32_t number_values)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
        initInvCrfKernel<<<block_count, block_size>>>(I, number_values);
    }
    cudaDeviceSynchronize();

    normInvCrf(I, number_values);

    // cudaDeviceSynchronize(); not needed
}

void normInvCrf(float* I, uint32_t number_values)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
        normInvCrfKernel<<<block_count, block_size>>>(I, number_values);
    }

    cudaDeviceSynchronize();
}

// Custom Stuff by the Students :D

void calcLightValsNumerator(float* x_num, uint8_t* pixels, float* weights, float* exposures, float* I, uint32_t number_imgs, uint32_t num_values, uint32_t iteration)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(num_values * number_imgs, block_size); // Spawn enough blocks
        calcLightValsNumeratorKernel<<<block_count, block_size>>>(x_num, pixels, weights, exposures, I, number_imgs, num_values, iteration);
    }

    cudaDeviceSynchronize();
}

void calcLightValsDenominator(float* x_denom, uint8_t* pixels, float* weights, float* exposures, uint32_t number_imgs, uint32_t num_values)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(num_values * number_imgs, block_size); // Spawn enough blocks
        calcLightValsDenominatorKernel<<<block_count, block_size>>>(x_denom, pixels, weights, exposures, number_imgs, num_values);
    }

    cudaDeviceSynchronize();
}

void calcLightValsDiv(float* x_hat, float* x_num, float* x_denom, uint32_t num_values)
{

    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(num_values, block_size); // Spawn enough blocks
        calcLightValsDivKernel<<<block_count, block_size>>>(x_hat, x_num, x_denom, num_values);
    }

    cudaDeviceSynchronize();
}

void calcIEstim(float* I_unnorm_buffer, float* exposures, float* x, uint8_t* pixels, bool* underexposed_mask, uint32_t* counters, uint32_t number_imgs, uint32_t num_values)
{
    // launch Sum kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(num_values*number_imgs, block_size); // Spawn enough blocks
        calcIEstimSumKernel<<<block_count, block_size>>>(I_unnorm_buffer, exposures, x, pixels, underexposed_mask, number_imgs, num_values);
    }

    cudaDeviceSynchronize();

    // launch Scale kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(256, block_size); // Spawn enough blocks
        calcIEstimScaleKernel<<<block_count, block_size>>>(I_unnorm_buffer, counters, 256);
    }

    cudaDeviceSynchronize();
}