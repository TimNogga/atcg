#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "photonmappingraygenerator.cuh"
#include "common.h"

#include "opg/hostdevice/ray.h"
#include "opg/hostdevice/color.h"
#include "opg/hostdevice/binarysearch.h"
#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"

__constant__ PhotonMappingLaunchParams params;

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
struct RayContextPhoton
{
    // Is the ray valid and should we continue tracing?
    bool        valid;
    // The ray that we are currently shooting through the scene.
    opg::Ray    ray;
    // How much is the radiance going to be attenuated between the current ray and the camera (based on all previous interactions).
    glm::vec3   throughput;
    // The depth of this ray, i.e. the number of bounces between the camera and this ray
    uint32_t    depth;
    // The initial weight of a photon when it was spawned from a light source.
    // This is used as reference for russian roulette since the weight is not neccessarily 1, depending on the brightness of the light sources in the scene...
    float initial_scalar_photon_weight;
};

// The context of a ray holds all relevant variables that need to "survive" each ray tracing iteration.
struct RayContextCamera
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


__forceinline__ __device__ float multiple_importance_weight(float this_pdf, float other_pdf)
{
    // Power heuristic with p=2
    //this_pdf *= this_pdf;
    //other_pdf *= other_pdf;
    return this_pdf / (this_pdf + other_pdf);
}


//
// Helper for storing photons
//

// Store a photon at the next available position in the photon map.
// Returns whether there the store operation was successfull.
__forceinline__ __device__ bool storePhoton(const glm::vec3 &photon_position, const glm::vec3 &photon_normal, const glm::vec3 &photon_irradiance_weight)
{
    uint32_t store_photon_index = atomicAdd(&params.photon_params.photon_store_count->desired_count, 1);
    uint32_t photon_map_size = params.photon_params.photon_positions.count;

    if (store_photon_index >= photon_map_size)
    {
        // This does not need to be atomic, if this condition is met, all threads will write the same value here!
        //*params.photon_params.photon_store_count = photon_map_size;
        // Storage unsuccessfull
        return false;
    }

    params.photon_params.photon_positions[store_photon_index] = photon_position;
    params.photon_params.photon_normals[store_photon_index] = photon_normal;
    params.photon_params.photon_irradiance_weights[store_photon_index] = photon_irradiance_weight;

    // Successfully stored *this* photon.
    return true;
}

__forceinline__ __device__ bool isPhotonMapFull()
{
    return params.photon_params.photon_store_count->desired_count >= params.photon_params.photon_positions.count;
}

__forceinline__ __device__ bool isPhotonMapAlmostFull()
{
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();
    uint32_t active_threads = launch_dims.x; // TODO count actually active threads!
    uint32_t expected_number_of_stores = 10; // Number of interactions we expect for each emitted photon.
    return params.photon_params.photon_store_count->desired_count + expected_number_of_stores * active_threads >= params.photon_params.photon_positions.count;
}


// Sample a new photon uniformly from all light sources in the scene.
// This implies that large light sources are sampled more frequently and brighter light sources are sampled more frequently.
// The relative weighting of the light sources is handled via the params.lights_cdf.
__device__ RayContextPhoton samplePhotonFromAllEmitters(PCG32 &rng)
{
    // The selected emitter
    const EmitterVPtrTable *emitter = params.emitters[0];
    // Probability of selecting this emitter
    float emitter_pdf = 1.0f;

    // Choose a light source using binary search in CDF values
    uint32_t emitter_index = opg::binary_search(params.emitters_cdf, rng.next1d());
    emitter = params.emitters[emitter_index];
    // Probability of selecting this emitter:
    emitter_pdf = params.emitters_cdf[emitter_index] - (emitter_index > 0 ? params.emitters_cdf[emitter_index-1] : 0.0f);
    EmitterPhotonSamplingResult sampling_result = emitter->samplePhoton(rng);

    RayContextPhoton photon_ctx;
    photon_ctx.valid = true;
    photon_ctx.ray.origin = sampling_result.position;
    photon_ctx.ray.direction = sampling_result.direction;
    photon_ctx.throughput = sampling_result.radiance_weight / emitter_pdf;
    photon_ctx.depth = 0;
    photon_ctx.initial_scalar_photon_weight = rgb_to_scalar_weight(photon_ctx.throughput);
    return photon_ctx;
}

extern "C" __global__ void __raygen__traceLights()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM();
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();

    uint32_t thread_index = launch_idx.x;

    // Initialize the random number generator.
    uint64_t seed = sampleTEA64(thread_index, params.subframe_index);
    PCG32 rng(seed);

    // Number of photons spawned for this launch
    uint32_t photon_emitted_count = 0;
    uint32_t photon_store_count_actual = 0;

    RayContextPhoton ray_ctx;
    ray_ctx.valid = false;

    while (!isPhotonMapFull())
    {
        // Spawn new photon
        if (!ray_ctx.valid)
        {
            // Don't start a new photon, if we expect to not store any more photons!
            if (isPhotonMapAlmostFull())
                break;
            ray_ctx = samplePhotonFromAllEmitters(rng);
            photon_emitted_count++;
        }

        // Russian roulette, boost the throughput of the photon back to 1, only do this if the photon weight is < 1!
        float scalar_photon_weight = rgb_to_scalar_weight(ray_ctx.throughput) / ray_ctx.initial_scalar_photon_weight;
        if (scalar_photon_weight < 1)
        {
            if (rng.next1d() < scalar_photon_weight)
            {
                // Divide throughput by the probability that the photon survives (does not get absorbed)
                ray_ctx.throughput /= scalar_photon_weight;
            }
            else
            {
                // This photon gets absorbed
                ray_ctx.valid = false;
                // Not break -> maybe start a new photon in the next iteration.
                continue;
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

        // Terminate this ray
        if (!si.is_finite())
        {
            ray_ctx.valid = false;
            // Not break -> maybe start a new photon in the next iteration.
            continue;
        }
        if (si.bsdf == nullptr)
        {
            ray_ctx.valid = false;
            continue;
        }

        // Deposit photon if this surface is diffuse
        if (has_flag(si.bsdf->component_flags, BSDFComponentFlag::DiffuseReflection))
        {
            glm::vec3 oriented_normal = glm::dot(si.incoming_ray_dir, si.normal) < 0 ? si.normal : -si.normal;
            // This quantity includes <N,L> term at the surface.
            // It does not explicitly appear here beacuse of the construction of the path from the "other side".
            glm::vec3 irradinace_weight = ray_ctx.throughput;
            if (storePhoton(si.position, oriented_normal, irradinace_weight))
            {
                ++photon_store_count_actual;
            }
        }

        // No further bounces for this surface interaction if ray depth is exceeded
        if (ray_ctx.depth >= MAX_TRACE_DEPTH)
        {
            ray_ctx.valid = false;
            // Not break -> maybe start a new photon in the next iteration.
            continue;
        }
        // Increase ray depth for recursive ray.
        ray_ctx.depth += 1;


        // Scatter photon, generate next ray
        BSDFSamplingResult bsdf_sampling_result = si.bsdf->sampleBSDF(si, +BSDFComponentFlag::Any, rng);
        if (bsdf_sampling_result.sampling_pdf == 0)
        {
            ray_ctx.valid = false;
            // Not break -> maybe start a new photon in the next iteration.
            continue;
        }
        ray_ctx.ray.origin = si.position;
        ray_ctx.ray.direction = bsdf_sampling_result.outgoing_ray_dir;
        ray_ctx.throughput *= bsdf_sampling_result.bsdf_weight;
    }

    // Accumulate number of photons emitted
    // Note that this is different from the number of photons stored in the photon map!
    atomicAdd(params.photon_params.photon_emitted_count, photon_emitted_count);
    atomicAdd(&params.photon_params.photon_store_count->actual_count, photon_store_count_actual);
}


extern "C" __global__ void __raygen__traceCamera()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM(); // Current Pixel coordinates
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM(); // Current Image dims

    auto &image_params = params.image_params;

    // Index of current pixel
    const glm::uvec2 pixel_index = glm::uvec2(launch_idx.x, launch_idx.y);
    const uint32_t linear_pixel_index = pixel_index.y * image_params.image_width + pixel_index.x;

    // Initialize the random number generator.
    // used to jitter the ray origin within a pixel for reducing aliasing
    uint64_t seed = sampleTEA64(linear_pixel_index, params.subframe_index);
    PCG32 rng(seed);

    RayContextCamera ray_ctx;
    {
        ray_ctx.valid = true;

        // Spawn the initial camera ray
        {
            // Choose uv coordinate uniformly at random within the pixel.
            glm::vec2 uv = (glm::vec2(pixel_index) + rng.next2d()) / glm::vec2(image_params.image_width, image_params.image_height);
            uv = 2.0f*uv - 1.0f; // [0, 1] -> [-1, 1]

            // generate a ray from the camer athrough the current pixel
            spawn_camera_ray(image_params.camera, uv, ray_ctx.ray.origin, ray_ctx.ray.direction);
        }

        // Initialize remainder of the context
        ray_ctx.throughput = glm::vec3(1); // stores contribution. Starts as white
        ray_ctx.depth = 0; // bounce amount (limit recursion)
        ray_ctx.last_sampling_pdf_for_mis = 1.0f; // Stores the probability density function (PDF) for the last BSDF sampling, used for MIS
        ray_ctx.last_interaction_for_mis.set_invalid();
        ray_ctx.output_radiance = glm::vec3(0); // accumulate rgb value for the pixel
    }

    // init storage variables as (0,0,0)
    glm::vec3 photon_gather_normal = glm::vec3(1);
    glm::vec3 photon_gather_throughput = glm::vec3(1);
    glm::vec3 photon_gather_position = glm::vec3(1);

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

        // Terminate this ray (it does not hit)
        if (!si.is_finite())
        {
            ray_ctx.valid = false;
            break;
        }

        
        // Uncomment to visualize (raw) photon map!
        // photon_gather_throughput = glm::vec3(1/glm::pi<float>() );
        // photon_gather_position = si.position;
        // photon_gather_normal = si.normal;
        // break;
        

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
            // Compute the MIS weight
            float mi_weight = multiple_importance_weight(bsdf_sampling_pdf, emitter_sampling_pdf);
            ray_ctx.output_radiance += ray_ctx.throughput * si.emitter->evalLight(si) * mi_weight;
        }

        if (si.bsdf == nullptr)
            continue;


        /* Implement:
         * - If the bsdf at the surface interaction is a diffuse BSDF, gather from photon map instead of continuing to trace rays.
         * Hint: Fill the photon_gather_* variables, the actual gathering from the photon map happens in a subsequent compute shader pass.
         */

        // TODO implement

        //TODO: Is this really it???

        bool isDiffuse = (si.bsdf->component_flags & BSDFComponentFlag::DiffuseReflection) || (si.bsdf->component_flags & BSDFComponentFlag::DiffuseTransmission);
        if (isDiffuse)
        {

            photon_gather_normal = si.normal;
            photon_gather_position = si.position;
            photon_gather_throughput = ray_ctx.throughput * si.bsdf->evalDiffuseAlbedo(si);

            ray_ctx.valid = false; // with this set only the diffuse part is rendered

            break;
        }

        //


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

            // Multiple importance sampling weight for BSDF and emitter importance sampling.
            float mi_weight = multiple_importance_weight(emitter_sampling_result.sampling_pdf, bsdf_sampling_pdf);

            ray_ctx.output_radiance += ray_ctx.throughput * bsdf_value * emitter_sampling_result.radiance_weight_at_receiver * mi_weight;
        }
        while (false);


        // No further indirect bounces for this surface interaction if ray depth is exceeded
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

    // Write per-pixel photon gather request
    image_params.photon_gather_normals(pixel_index).value() = photon_gather_normal;
    image_params.photon_gather_throughputs(pixel_index).value() = photon_gather_throughput;
    image_params.photon_gather_positions(pixel_index).value() = photon_gather_position;

    // Write linear output color
    image_params.output_radiance(pixel_index).value() = ray_ctx.output_radiance;
}
