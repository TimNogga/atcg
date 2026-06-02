#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "pathtracingraygenerator.cuh"

#include "opg/hostdevice/ray.h"
#include "opg/hostdevice/color.h"
#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"

__constant__ PathtracingLaunchParams params;

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

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}

// The context of a ray holds all relevant variables that need to "survive" each ray tracing iteration.
struct RayContext
{
    // Is the ray valid and should we continue tracing?
    bool        valid;
    // The ray that we are currently shooting through the scene.
    opg::Ray    ray;
    // How much is the radiance going to be attenuated between the current ray and the camera (based on all previous interactions).
    glm::vec3   throughput;
    // The depth of this ray, i.e. the number of bounces between the camera and this ray
    uint32_t    depth;

    // Radiance accumulated along the current ray
    glm::vec3 output_radiance;
};


extern "C" __global__ void __raygen__main()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM();
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();

    // Index of current pixel
    const glm::uvec2 pixel_index = glm::uvec2(launch_idx.x, launch_idx.y);
    const uint32_t linear_pixel_index = pixel_index.y * params.image_width + pixel_index.x;

    // Initialize the random number generator.
    uint64_t seed = sampleTEA64(linear_pixel_index, params.subframe_index);
    PCG32 rng = PCG32(seed);

    // Initialize the ray context.
    RayContext ray_ctx;
    {
        ray_ctx.valid = true;

        // Spawn the initial camera ray
        {
            // Choose uv coordinate uniformly at random within the pixel.
            glm::vec2 uv = (glm::vec2(pixel_index) + rng.next2d()) / glm::vec2(params.image_width, params.image_height);
            uv = 2.0f*uv - 1.0f; // [0, 1] -> [-1, 1]

            spawn_camera_ray(params.camera, uv, ray_ctx.ray.origin, ray_ctx.ray.direction);
        }

        // Initialize remainder of the context
        ray_ctx.throughput = glm::vec3(1);
        ray_ctx.depth = 0;
        ray_ctx.output_radiance = glm::vec3(0);
    }


    // Bound the number of trace operations that are executed in any case.
    // Accidentally creating infinite loops on the GPU is unpleasant.
    for (int i = 0; i < MAX_TRACE_OPS; ++i)
    {
        if (!ray_ctx.valid)
            break;

        // Trace current ray
        SurfaceInteraction si;
        traceWithDataPointer<SurfaceInteraction>(
                params.traversable_handle,
                ray_ctx.ray.origin,
                ray_ctx.ray.direction,
                params.scene_epsilon,                   // tmin: Start ray at ray_origin + tmin * ray_direction
                std::numeric_limits<float>::infinity(), // tmax: End ray at ray_origin + tmax * ray_direction
                params.surface_interaction_trace_params,
                &si
        );

        // Terminate this ray if no valid surface interaction was found.
        if (!si.is_valid() || si.bsdf == nullptr)
        {
            ray_ctx.valid = false;
            break;
        }

        // Direct illumination for each emitter
        for (uint32_t emitter_index = 0; emitter_index < params.emitters.count; ++emitter_index)
        {
            const EmitterVPtrTable *emitter = params.emitters[emitter_index];
            EmitterSamplingResult emitter_sampling_result = emitter->sampleLight(si, rng);

            opg::Ray ray_to_emitter;
            ray_to_emitter.origin = si.position;
            ray_to_emitter.direction = emitter_sampling_result.direction_to_light;

            // Cast shadow ray
            bool occluded = traceOcclusion(
                    params.traversable_handle,
                    ray_to_emitter.origin,
                    ray_to_emitter.direction,
                    params.scene_epsilon,
                    emitter_sampling_result.distance_to_light - params.scene_epsilon,
                    params.occlusion_trace_params
            );
            if (occluded)
                continue;

            BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling_result.direction_to_light, +BSDFComponentFlag::Any);
            ray_ctx.output_radiance += ray_ctx.throughput * bsdf_result.bsdf_value * emitter_sampling_result.radiance_weight_at_receiver;
        }

        // No further indirect bounces if ray depth is exceeded
        if (ray_ctx.depth >= MAX_TRACE_DEPTH)
        {
            ray_ctx.valid = false;
            break;
        }
        // Increase ray depth for recursive ray.
        ray_ctx.depth += 1;

        //
        // Indirect illumination, generate next ray
        //
        BSDFSamplingResult bsdf_sampling_result = si.bsdf->sampleBSDF(si, +BSDFComponentFlag::Any, rng);
        if (bsdf_sampling_result.sampling_pdf == 0)
        {
            ray_ctx.valid = false;
            break;
        }

        // Construct the next ray
        ray_ctx.ray.direction = glm::normalize(bsdf_sampling_result.outgoing_ray_dir);
        ray_ctx.ray.origin = si.position;
        ray_ctx.throughput *= bsdf_sampling_result.bsdf_weight;

        // Check if the sampled ray is inconsistent wrt. geometric and shading normal...
        if (glm::dot(ray_ctx.ray.direction, si.geom_normal) * glm::dot(ray_ctx.ray.direction, si.normal) <= 0)
        {
            ray_ctx.valid = false;
            break;
        }
    }

    //
    // Update results
    //

    // Write linear output color
    params.output_radiance(pixel_index).value() = ray_ctx.output_radiance;
}
