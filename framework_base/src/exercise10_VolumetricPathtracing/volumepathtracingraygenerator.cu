#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "volumepathtracingraygenerator.cuh"

#include "opg/hostdevice/ray.h"
#include "opg/hostdevice/color.h"
#include "opg/hostdevice/binarysearch.h"
#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"

__constant__ VolumePathtracingLaunchParams params;

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
    // The medium that the ray travels through
    const MediumVPtrTable *medium;

    // Radiance accumulated along the current ray
    glm::vec3 output_radiance;
};


__forceinline__ __device__ float multiple_importance_weight(float this_pdf, float other_pdf)
{
    // Power heuristic with p=2
    //this_pdf *= this_pdf;
    //other_pdf *= other_pdf;
    //return this_pdf / (this_pdf + other_pdf);

    // Balance heuristic with p=1
    return this_pdf / (this_pdf + other_pdf);

    // No mis with p=0
    // Note that the weight is not 1, but 0.5 if no multiple importance sampling is done with two sampling strategies!
    // This can be interpreted as if half of the contribution is integrated using one sampling strategy,
    // and the other half of the contribution is integrated using the other sampling strategy.
    // If the other sampling strategy cannot produce the sample, it is only integrated using this sampling strategy.
    return other_pdf == 0.0f ? 1.0f : 0.5f;
}


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
        ray_ctx.medium = params.initial_camera_medium;
        ray_ctx.output_radiance = glm::vec3(0);
    }

    // Bound the number of trace operations that are executed in any case.
    // Accidentally creating infinite loops on the GPU is unpleasant.
    for (int i = 0; i < MAX_TRACE_OPS; ++i)
    {
        if (!ray_ctx.valid)
            break;

        // Russian roulette
        float throughput_weight = rgb_to_scalar_weight(ray_ctx.throughput);
        if (throughput_weight < 1)
        {
            if (rng.next1d() < throughput_weight)
            {
                // Boost this ray.
                ray_ctx.throughput /= throughput_weight;
            }
            else
            {
                // Terminate this ray.
                ray_ctx.valid = false;
                break;
            }
        }

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

        // Apply medium attenuation along the ray, maybe sample medium scattering event!!!
        if (ray_ctx.medium != nullptr)
        {
            // TODO max_distance from si.incoming_distance?
            float max_distance = si.is_finite() ? glm::length(si.position - ray_ctx.ray.origin) - params.scene_epsilon : std::numeric_limits<float>::infinity();
            MediumSamplingResult result = ray_ctx.medium->sampleMediumEvent(ray_ctx.ray, max_distance, rng);
            // Attenuate the ray throughput in case of medium event *and* no medium event.
            ray_ctx.throughput *= result.transmittance_weight;
            // Handle medium or no-medium event.
            if (!result.interaction.is_finite())
            {
                // No medium event.
                // Nothing, continue below with surface event handling.
            }
            else
            {
                // Medium event!
                // Importance-sample phase function and next event estimation to light source
                const PhaseFunctionVPtrTable *phase_function = ray_ctx.medium->phase_function;

                // The medium interaction we found here
                MediumInteraction mi = result.interaction;

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
                    EmitterSamplingResult emitter_sampling_result = emitter->sampleLight(mi, rng);

                    if (emitter_sampling_result.sampling_pdf == 0)
                    {
                        // Sampling failed...
                        continue;
                    }
                    // Account for emitter selection pdf
                    emitter_sampling_result.radiance_weight_at_receiver /= emitter_selection_pdf;
                    emitter_sampling_result.sampling_pdf *= emitter_selection_pdf;

                    opg::Ray ray_to_emitter;
                    ray_to_emitter.origin = mi.position;
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

                    // No medium event before light source
                    // This is an ugly workaround.
                    // We use the incoming_ray_dir of a medium event as the forward direction for evaluating the transmittance in the interface.
                    // However, the original incoming_ray_dir of the medium event is not the forward direction towards the light source!
                    glm::vec3 medium_transmittance = ray_ctx.medium->evalTransmittance(ray_to_emitter, emitter_sampling_result.distance_to_light, rng);

                    // Evaluate phase function and probability of generating this ray via phase function importance sampling.
                    PhaseFunctionEvalResult phase_result = phase_function->evalPhaseFunction(mi, emitter_sampling_result.direction_to_light);
                    glm::vec3 phase_function_value = phase_result.phase_function_value;
                    float phase_sampling_pdf = phase_result.sampling_pdf;

                    // Multiple importance sampling weight for phase function and emitter importance sampling.
                    float mi_weight = multiple_importance_weight(emitter_sampling_result.sampling_pdf, phase_sampling_pdf);

                    ray_ctx.output_radiance += ray_ctx.throughput * medium_transmittance * phase_function_value * emitter_sampling_result.radiance_weight_at_receiver * mi_weight;
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


                // Sample the phase function at the medium interaction
                PhaseFunctionSamplingResult phase_fun_sampling_result = phase_function->samplePhaseFunction(mi, rng);
                if (phase_fun_sampling_result.sampling_pdf == 0)
                {
                    ray_ctx.valid = false;
                    break;
                }

                // Construct the next ray
                // When considering the phase-function equivalent of BSSRDFs, the position might change again!
                ray_ctx.ray.direction = glm::normalize(phase_fun_sampling_result.outgoing_ray_dir);
                // Next ray starts at medium event, not surface event!
                ray_ctx.ray.origin = mi.position;
                ray_ctx.throughput *= phase_fun_sampling_result.phase_function_weight;

                // The sampling pdf for mis
                ray_ctx.last_sampling_pdf_for_mis = phase_fun_sampling_result.sampling_pdf;
                // The previous interaction is the sampled medium interaction.
                ray_ctx.last_interaction_for_mis = mi;

                // Skip the processing of BSDF and emitter on the surface interaction below, since the surface was not reached!
                continue;
            }
        }

        // Terminate this ray if no medium event (code above) and no surface interaction (code below)
        if (!si.is_finite())
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
            // Probability of sampling the ray that results in this surface interaction via light source sampling
            float emitter_selection_pdf = si.emitter->emitter_weight / params.emitters_total_weight;
            float emitter_sampling_pdf = can_ray_be_generated_by_light_source_sampling ? emitter_selection_pdf * si.emitter->evalLightSamplingPdf(ray_ctx.last_interaction_for_mis, si) : 0;
            // Probability of sampling this ray via BSDF importance sampling (computed at the end of the previous iteration of the main loop)
            float bsdf_sampling_pdf = ray_ctx.last_sampling_pdf_for_mis;

            // Multiple importance sampling weight for BSDF and emitter importance sampling
            float mi_weight = multiple_importance_weight(bsdf_sampling_pdf, emitter_sampling_pdf);

            ray_ctx.output_radiance += ray_ctx.throughput * si.emitter->evalLight(si) * mi_weight;
        }

        if (si.bsdf == nullptr)
        {
            ray_ctx.valid = false;
            break;
        }

        //
        // Next event estimation towards emitters in scene
        //

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

            glm::vec3 medium_transmittance = glm::vec3(1);
            // If there is a medium present, it will attenuate the light between the light source and the current surface interaction.
            if (ray_ctx.medium != nullptr)
            {
                // Probability of no medium event along path.
                medium_transmittance = ray_ctx.medium->evalTransmittance(ray_to_emitter, emitter_sampling_result.distance_to_light, rng);
            }

            // Evaluate BSDf and probability of generating this ray via BSDF importance sampling.
            BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling_result.direction_to_light, +BSDFComponentFlag::Any);
            glm::vec3 bsdf_value = bsdf_result.bsdf_value;
            float bsdf_sampling_pdf = bsdf_result.sampling_pdf;

            // Multiple importance sampling weight for BSDF and emitter importance sampling.
            float mi_weight = multiple_importance_weight(emitter_sampling_result.sampling_pdf, bsdf_sampling_pdf);

            ray_ctx.output_radiance += ray_ctx.throughput * medium_transmittance * bsdf_value * emitter_sampling_result.radiance_weight_at_receiver * mi_weight;
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
        // Move ray origin away from surface interaction!
        ray_ctx.ray.origin = si.position + params.scene_epsilon * glm::sign(glm::dot(ray_ctx.ray.direction, si.normal)) * si.normal;
        ray_ctx.throughput *= bsdf_sampling_result.bsdf_weight;

        // Check if the sampled ray is inconsistent wrt. geometric and shading normal...
        if (glm::dot(ray_ctx.ray.direction, si.geom_normal) * glm::dot(ray_ctx.ray.direction, si.normal) <= 0)
        {
            ray_ctx.valid = false;
            break;
        }

        // Check if we are entering the geometry or leaving the geometry and assign si.inside_medium or si.outside_medium, respectively.
        float cos_theta_curr_ray = glm::dot(si.geom_normal, si.incoming_ray_dir);
        float cos_theta_next_ray = glm::dot(si.geom_normal, bsdf_sampling_result.outgoing_ray_dir);
        // Only change the medium if we have a transmission...
        if (cos_theta_curr_ray*cos_theta_next_ray > 0)
        {
            ray_ctx.medium = cos_theta_next_ray < 0 ? si.inside_medium : si.outside_medium;
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
