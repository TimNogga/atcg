#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>

struct PathtracingLaunchParams
{
    // Small floating point number used to move rays away from surfaces when tracing to avoid self intersections.
    float scene_epsilon;

    // The index of the current frame is used to generate a different random seed for each frame
    uint32_t      subframe_index;

    // 2D output buffer storing the final linear color values for the current subframe only.
    // Each thread in the __raygen__main shader program computes one pixel value in this array.
    // Access via output_buffer[y][x].value() or output_buffer(glm::uvec2(x, y)).value()
    opg::TensorView<glm::vec3, 2> output_radiance;

    // Size of the image we are generating
    uint32_t image_width;
    uint32_t image_height;

    // Camera parameters that are used to spawn rays through the pixels of the image
    CameraData camera;

    // Sum over the weight of all emitters in the scene (Used to compute the emitter selection probability)
    float                                     emitters_total_weight;
    // The cummulative distribution over all emitters in the scene (Used to select an emitter during next-event estimation)
    opg::BufferView<float>                    emitters_cdf;
    // List of all emitters in the scene
    opg::BufferView<const EmitterVPtrTable *> emitters;

    // Additional parameters used to initiate ray tracing operations
    TraceParameters surface_interaction_trace_params;
    TraceParameters occlusion_trace_params;
    OptixTraversableHandle traversable_handle; // top-level scene IAS
};

#define MAX_TRACE_OPS   (300)
#define MAX_TRACE_DEPTH (8)
