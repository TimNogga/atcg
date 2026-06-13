#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "pathtracingraygenerator.cuh"

#include "opg/hostdevice/ray.h"
#include "opg/hostdevice/color.h"
#include "opg/hostdevice/binarysearch.h"
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
    // PDF of selecting this ray via BSDF or phase function sampling given the previous ray...
    float       last_sampling_pdf_for_mis;
    // The interaction from which the current ray was sampled (if valid)
    Interaction last_interaction_for_mis;

    // Radiance accumulated along the current ray
    glm::vec3 output_radiance;
};


//


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
        ray_ctx.last_sampling_pdf_for_mis = 1.0f;
        ray_ctx.last_interaction_for_mis.set_invalid();
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
        if (!si.is_valid())
        {
            ray_ctx.valid = false;
            break;
        }

        // Handle emitter contribution if present
        if (si.emitter != nullptr)
        {
            // Multiple importance sampling
            // This ray could be generated from light source sampling if:
            // - the previous surface interaction is valid
            // - the previous surface interaction has a bsdf
            // - the bsdf is not an ideal reflection/transmission
            bool can_ray_be_generated_by_light_source_sampling = ray_ctx.last_interaction_for_mis.is_finite();

            /* Implement:
             * - Multiple importance sampling with the balance heuristic using emitter and BSDF importance sampling.
             * Hints:
             * - The BSDF sampling PDF that resulted in this interaction is stored in `ray_ctx.last_sampling_pdf_for_mis`.
             * - For the emitter sampling PDF, you can use the `EmitterVPtrTable::evalLightSamplingPdf()` function.
             * - Don't forget the probability of selecting an emitter for emitter importance sampling below.
             */

            // Multiple importance sampling weight
            float mi_weight = 1;

            // TODO implement
            if (can_ray_be_generated_by_light_source_sampling)
            {
                float bsdf_sampling_pdf = ray_ctx.last_sampling_pdf_for_mis;
                float emitter_selection_pdf = si.emitter->emitter_weight / params.emitters_total_weight;
                float light_sampling_pdf = emitter_selection_pdf * si.emitter->evalLightSamplingPdf(ray_ctx.last_interaction_for_mis, si);
                mi_weight = bsdf_sampling_pdf / (bsdf_sampling_pdf + light_sampling_pdf);
            }

            //

            ray_ctx.output_radiance += ray_ctx.throughput * si.emitter->evalLight(si) * mi_weight;
        }

        // Terminate this ray if no BSDF is present.
        if (si.bsdf == nullptr)
        {
            ray_ctx.valid = false;
            break;
        }

        // Next event estimation towards emitters in scene
        // do {...} while (false); loop with only **one** iteration is used such that we can "abort" the code block using the continue or break statements!
        do
        {
            // Choose a light source using binary search in CDF values
            uint32_t emitter_index = opg::binary_search(params.emitters_cdf, rng.next1d());
            const EmitterVPtrTable *emitter = params.emitters[emitter_index];
            // Probability of selecting this emitter:
            float emitter_selection_pdf = emitter->emitter_weight / params.emitters_total_weight;
            //float emitter_selection_pdf = params.emitters_cdf[emitter_index] - (emitter_index > 0 ? params.emitters_cdf[emitter_index-1] : 0.0f);
            EmitterSamplingResult emitter_sampling_result = emitter->sampleLight(si, rng);

            if (emitter_sampling_result.sampling_pdf == 0)
            {
                // Sampling failed...
                continue;
            }
            // Account for emitter selection pdf
            emitter_sampling_result.radiance_weight_at_receiver /= emitter_selection_pdf;
            emitter_sampling_result.sampling_pdf *= emitter_selection_pdf;

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

            // Evaluate BSDf and probability of generating this ray via BSDF importance sampling.
            BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling_result.direction_to_light, +BSDFComponentFlag::Any);
            glm::vec3 bsdf_value = bsdf_result.bsdf_value;
            float bsdf_sampling_pdf = bsdf_result.sampling_pdf;

            /* Implement:
             * - Multiple importance sampling weight using the balance heuristic using emitter and BSDF importance sampling.
             */

            // Multiple importance sampling weight
            float mi_weight = 1;

            // TODO implement
            mi_weight = emitter_sampling_result.sampling_pdf / (emitter_sampling_result.sampling_pdf + bsdf_sampling_pdf);

            //

            ray_ctx.output_radiance += ray_ctx.throughput * bsdf_value * emitter_sampling_result.radiance_weight_at_receiver * mi_weight;
        }
        while (false);

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

        // For MIS
        ray_ctx.last_sampling_pdf_for_mis = bsdf_sampling_result.sampling_pdf;
        // Remember the current surface interaction as "previous" surface interaction in the next iteration of the algorithm
        // The previous si is useful if we hit a light source (which we don't know at this point), and want to do multiple importance sampling
        // Then we want to query the probability of sampling a ray from this surface interaction (then previous_si) to the light source (at the then current surface interaction).
        ray_ctx.last_interaction_for_mis = si;
        // The interaction is only valid for MIS if the BSDF is not an ideal reflection or ideal transmission where light source sampling does not apply.
        if (has_flag(si.bsdf->component_flags, BSDFComponentFlag::AnyDelta))
            ray_ctx.last_interaction_for_mis.set_invalid();
    }

    //
    // Update results
    //

    // Write linear output color
    params.output_radiance(pixel_index).value() = ray_ctx.output_radiance;
}
