#pragma once

#include "opg/scene/interface/medium.cuh"

#include "opg/scene/components/grid.cuh"

struct HeterogeneousMediumData
{
    // The heterogeneous volume data spans the [0, 1]^3 unit cube in a local coordinate system
    glm::mat4 world_to_local;
    // The heterogeneous volume data
    GridData<float> density_grid;
    // A scaling factor that is applied to the contents of density_data.
    float density_scale;
    // The maximum density value found in the already scaled density data.
    float density_majorant;
};
