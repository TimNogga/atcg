#pragma once

#include "opg/glmwrapper.h"


//
// Sampling functions
//

__forceinline__ __device__ glm::vec3 warp_square_to_sphere_uniform(const glm::vec2 uv)
{
    float z   = uv.x * 2 - 1;
    float phi = uv.y * 2 * glm::pi<float>();

    float r = glm::sqrt( glm::max(0.0f, 1 - z*z) );
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_sphere_uniform_pdf(const glm::vec3 &dir)
{
    return 1 / (4 * glm::pi<float>());
}


__forceinline__ __device__ glm::vec3 warp_square_to_spherical_cap_uniform(const glm::vec2 &uv, float cap_height)
{
    // See https://en.wikipedia.org/wiki/Spherical_cap

    float z = glm::lerp(1.0f-cap_height, 1.0f, uv.x);
    float phi = 2*glm::pi<float>() * uv.y;
    float r = glm::sqrt(1 - z*z);
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_spherical_cap_uniform_pdf(const glm::vec3 &dir, float cap_height)
{
    return 1/(2 * glm::pi<float>() * cap_height);
}


__forceinline__ __device__ glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2 &uv)
{
    float r_sq = uv.x;
    float r   = glm::sqrt(r_sq);
    float phi = 2*glm::pi<float>() * uv.y;

    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);
    float z = glm::sqrt(glm::max(0.0f, 1.0f - r_sq));

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_hemisphere_cosine_pdf(const glm::vec3 &result)
{
    return glm::max(0.0f, result.z) / glm::pi<float>();
}
