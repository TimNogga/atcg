#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/trace.cuh"

#include "photonmapdata.cuh"

#include <optix.h>

struct TracePhotonsOnlyParams
{
    // Buffers holding the attributes of generated photons
    opg::BufferView<glm::vec3>    photon_positions;
    opg::BufferView<glm::vec3>    photon_normals;
    opg::BufferView<glm::vec3>    photon_irradiance_weights;

    // Count how many photons have been stored in the photon map.
    // The value stored at this address is incremented by all threads when they generate a photon by calling atomicAdd().
    // It is used to determine the location of new photons and detect when the photon map is full.
    PhotonMapStoreCount*          photon_store_count;

    // The number of photons that have been spawned from a light source by each thread
    uint32_t*                     photon_emitted_count;
};

struct TraceImageOnlyParams
{
    // Additional output buffers, storing one value for each pixel:
    // For each pixel we store the world-space position and normal of the surface for which we want to gather photons, as well as the weight/throughput for the pixel value.
    opg::TensorView<glm::vec3, 2>   photon_gather_positions;
    opg::TensorView<glm::vec3, 2>   photon_gather_normals;
    opg::TensorView<glm::vec3, 2>   photon_gather_throughputs;

    // 2D output buffer storing the final linear color values for the current subframe only.
    // Each thread in the __raygen__main shader program computes one pixel value in this array.
    // Access via output_buffer[y][x].value() or output_buffer(glm::uvec2(x, y)).value()
    opg::TensorView<glm::vec3, 2>   output_radiance;

    // Size of the image we are generating
    uint32_t image_width;
    uint32_t image_height;

    CameraData camera;
};

struct PhotonMappingLaunchParams
{
    // Small floating point number used to move rays away from surfaces when tracing to avoid self intersections.
    float scene_epsilon;

    // The index of the current frame is used to generate a different random seed for each frame.
    // In case we are using multiple launches to trace a lot of photons, the subframe_index is used to create a different seed each time.
    uint32_t      subframe_index;

    // Make the memory of `photon_params` and `image_params` alias, i.e. they overlap and use the same memory.
    union {
        TracePhotonsOnlyParams photon_params;
        TraceImageOnlyParams image_params;
    };

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

    // Explicit empty constructor...
    PhotonMappingLaunchParams() {};
};

#define MAX_TRACE_OPS   (300)
#define MAX_TRACE_DEPTH (8)
