#pragma once

#include "opg/scene/interface/bsdf.cuh"

struct DiffuseBSDFData
{
    glm::vec3 diffuse_color;
};

struct SpecularBSDFData
{
    glm::vec3 specular_F0;
};

struct RefractiveBSDFData
{
    float index_of_refraction;
};
