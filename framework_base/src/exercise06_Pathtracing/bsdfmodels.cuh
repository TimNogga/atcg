#pragma once

#include "opg/scene/interface/bsdf.cuh"

struct GGXBSDFData
{
    glm::vec3 diffuse_color;
    glm::vec3 specular_F0;
    float roughness;
};

struct RefractiveBSDFData
{
    float index_of_refraction;
};
