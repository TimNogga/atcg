#pragma once

#include "opg/scene/interface/phasefunction.cuh"

struct HenyeyGreensteinPhaseFunctionData
{
    // The HG phase function is parameterized by the mean dot(incomming_ray_dir, outgoing_ray_dir)
    float g;
};
