#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

#include "opg/glmwrapper.h"

// forward declaration
namespace opg {
    struct ImageData;
}

void splitChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels);
void splitChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels);
void mergeChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels);
void mergeChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels);

void calcMask(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image);
void countValues(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image);
void calcWeights(uint8_t* values, float* weights, uint32_t number_values);

void initInvCrf(float* I, uint32_t number_values);
void normInvCrf(float* I, uint32_t number_values);

// TODO: declare your functions here
//

void calcLightValsNumerator(float* x_num, uint8_t* pixels, float* weights, float* exposures, float* I, uint32_t number_imgs, uint32_t num_values, uint32_t iteration);
void calcLightValsDenominator(float* x_denom, uint8_t* pixels, float* weights, float* exposures, uint32_t number_imgs, uint32_t num_values);
void calcLightValsDiv(float* x_hat, float* x_num, float* x_denom, uint32_t num_values);

void calcIEstim(float* I_unnorm_buffer, float* exposures, float* x, uint8_t* pixels, bool* underexposed_mask, uint32_t* counters, uint32_t number_imgs, uint32_t num_values);

std::pair<opg::ImageData, std::vector<std::vector<float>>> robertson(const std::vector<opg::ImageData> &imgs, const std::vector<float> &exposures, size_t max_iterations = 10);
