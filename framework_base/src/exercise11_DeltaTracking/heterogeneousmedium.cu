#include "heterogeneousmedium.cuh"
#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/color.h"

__forceinline__ __device__ float warp_1d_sample_to_homogeneous_medium_event_distance(float u, float sigma_t)
{
    float t = -glm::log(u) / sigma_t;
    return t;
}

__forceinline__ __device__ float warp_1d_sample_to_homogeneous_medium_event_distance_pdf(float t, float sigma_t)
{
    float pdf = sigma_t * glm::exp(-sigma_t * t);
    return pdf;
}

__device__ float evaluate_density_grid(glm::vec3 position)
{
    const HeterogeneousMediumData *sbt_data = *reinterpret_cast<const HeterogeneousMediumData **>(optixGetSbtDataPointer());
    glm::vec3 local_position = glm::vec3(sbt_data->world_to_local * glm::vec4(position, 1));
    float density = sbt_data->density_grid.eval(local_position);
    density *= sbt_data->density_scale;
    return density;
}

__device__ float sample_free_flight_distance_delta_tracking(const opg::Ray &ray, float max_distance, PCG32 &rng)
{
    const HeterogeneousMediumData *sbt_data = *reinterpret_cast<const HeterogeneousMediumData **>(optixGetSbtDataPointer());

    // Dummy implementation: Invalid distance, i.e. scattering event outside of the [0, max_distance] interval.
    float distance = std::numeric_limits<float>::signaling_NaN(); // NOTE: we don't care about signaling_NaN vs quiet_NaN...

    /* Implement:
     * - Sample the distance of a medium event in the inverval [0, max_distance] along the given ray in the medium using the delta-tracking algorithm.
     * Hint: Use the functions above to sample the medium density and sample distances in homogeneous media.
     */

    //

    return distance;
}

//

__device__ float estimate_transmittance(const opg::Ray &ray, float max_distance, PCG32 &rng)
{
    /* Implement:
     * - Evaluate the transmittance over a given distance along the ray, i.e. the transmittance between origin and origin+max_distance*direction.
     * - Implement either the delta-tracking based algorithm or ratio-tracking algorithm.
     * Hint: Use the functions above to sample the medium density and sample distances in homogeneous media.
     */

    //

    return 1;
}

extern "C" __device__ glm::vec3 __direct_callable__heterogeneousMedium_evalTransmittance(const opg::Ray &ray, float distance, PCG32 &rng)
{
    return glm::vec3(estimate_transmittance(ray, distance, rng));
}

extern "C" __device__ MediumSamplingResult __direct_callable__heterogeneousMedium_sampleMediumEvent(const opg::Ray &ray, float max_distance, PCG32 &rng)
{
    // Arbitrarily clamp max_distance.
    // If max_distance would be (close to) infinite, the loop below might not terminate.
    max_distance = glm::clamp(max_distance, 0.0f, 1e6f);

    float sampled_distance = sample_free_flight_distance_delta_tracking(ray, max_distance, rng);

    MediumSamplingResult result;
    // Set incoming ray direction
    result.interaction.incoming_ray_dir = ray.direction;
    result.interaction.incoming_distance = sampled_distance;
    result.transmittance_weight = glm::vec3(1);
    if (result.interaction.is_valid())
    {
        result.interaction.position = ray.at(result.interaction.incoming_distance);
    }
    return result;
}
