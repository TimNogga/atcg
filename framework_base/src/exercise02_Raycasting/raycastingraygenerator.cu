#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "raycastingraygenerator.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/hostdevice/color.h"

__constant__ RayCastingLaunchParams params;

extern "C" __global__ void __miss__main()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();

    si->incoming_ray_dir = world_ray_dir;

    // No valid interaction found, set incoming_distance to NaN
    si->set_invalid();
}

extern "C" __global__ void __raygen__main()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM();
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();

    // Index of current pixel in image
    const glm::uvec2 pixel_index = glm::uvec2(launch_idx.x, launch_idx.y);
    glm::vec2 viewport_coordinates = 2.0f * ((glm::vec2(pixel_index) + 0.5f) / glm::vec2(launch_dims)) - 1.0f;
    glm::vec3 ray_origin;
    glm::vec3 ray_dir;
    spawn_camera_ray(params.camera, viewport_coordinates, ray_origin, ray_dir);


    SurfaceInteraction si;
    traceWithDataPointer<SurfaceInteraction>(
            params.traversable_handle,
            ray_origin,
            ray_dir,
            0.0f,                                   // tmin: Start ray at ray_origin + tmin * ray_direction
            std::numeric_limits<float>::infinity(), // tmax: End ray at ray_origin + tmax * ray_direction
            params.traceParams,
            &si
    );

    glm::vec3 result = si.normal * 0.5f + 0.5f;
    if (!si.is_finite()) result = glm::vec3(0);

    // Write linear output color, want to interpret the result as "perceptually" linear colors with sRGB gamma applied.
    params.output_buffer(pixel_index).value() = apply_inverse_srgb_gamma(glm::clamp( result, 0.0f, 1.0f ));
}
