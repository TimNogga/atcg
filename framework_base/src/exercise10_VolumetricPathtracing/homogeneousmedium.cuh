#pragma once

#include "opg/scene/interface/medium.cuh"

struct HomogeneousMediumData
{
    // The absorbtion coefficient defines the probability per distance traveled of light being absorbed by the medium.
    glm::vec3 sigma_a;
    // The scattering coefficient defines the probability per distance traveled of light being scattered by the medium.
    glm::vec3 sigma_s;

    // The transmission coefficient that defines the density of the medium (sigma_s + sigma_a)
    // float sigma_t;
    // The albedo describes the ratio of scattering vs. absorbing particles in the medium
    // The scattering coefficient (sigma_s) is sigma_t * albedo, and the absorbtion coefficient (sigma_a) is sigma_t * (1-albedo)
    // glm::vec3 albedo;
};
