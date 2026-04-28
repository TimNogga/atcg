#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/memory/tensorview.h"

#include <optix.h>

struct RayCastingLaunchParams
{
    // 2D output buffer storing the final linear color values.
    // Each thread in the __raygen__main shader program computes one pixel value in this array.
    // Access via output_buffer[y][x].value() or output_buffer(glm::uvec2(x, y)).value()
    opg::TensorView<glm::vec3, 2> output_buffer;

    // Size of the image we are generating
    uint32_t image_width;
    uint32_t image_height;

    CameraData camera;

    TraceParameters traceParams;
    OptixTraversableHandle traversable_handle; // top-level scene IAS
};
