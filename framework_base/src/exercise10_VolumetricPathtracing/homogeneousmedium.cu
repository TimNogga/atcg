#include "homogeneousmedium.cuh"

#include "opg/hostdevice/color.h"


//


extern "C" __device__ glm::vec3 __direct_callable__homogeneousMedium_evalTransmittance(const opg::Ray &ray, float distance, PCG32 &unused_rng)
{
    const HomogeneousMediumData *sbt_data = *reinterpret_cast<const HomogeneousMediumData **>(optixGetSbtDataPointer());

    /* Implement:
     * - Evaluate the transmittance for a certain distance along the given ray, i.e. the probability of encountering no absorbtion and no out-scattering.
     */

    //

    // Dummy implementation with no attenuation.
    return glm::vec3(1);
}

extern "C"  __device__ MediumSamplingResult __direct_callable__homogeneousMedium_sampleMediumEvent(const opg::Ray &ray, float max_distance, PCG32 &rng)
{
    const HomogeneousMediumData *sbt_data = *reinterpret_cast<const HomogeneousMediumData **>(optixGetSbtDataPointer());

    // Absorbtion, scattering and extinction coefficients...
    glm::vec3 sigma_a = sbt_data->sigma_a;
    glm::vec3 sigma_s = sbt_data->sigma_s;
    glm::vec3 sigma_t = sigma_a + sigma_s;

    // Scalar projection of scattering coefficient, used to sample the next medium scattering event.
    float sigma_s_scalar = rgb_to_scalar_weight(sbt_data->sigma_s);


    MediumSamplingResult result;
    // Dummy implementation:
    // Effectively no medium event.
    result.interaction = MediumInteraction::empty();
    result.interaction.incoming_ray_dir = ray.direction;
    result.transmittance_weight = glm::vec3(1);

    /* Implement:
     * - Sample the free-flight distance that the ray travels through the medium without being scattered.
     *     Hint: If the distance is beyond the max_distance, create an invalid interaction using `result.interaction.set_invalid()`.
     * - Compute the respective sampling probability.
     * - Compute the transmittance along that distance, i.e. the probability of no outscattering and no absorbtion,
     *   including the scattering coefficient, i.e. the density of encountering a scattering event *at* the sampled distance.
     */

    // If the scattering coefficient is 0, we don't need to sample a medium event where the phase function is then evaulated.
    // To handle the case of absorbtion we still need to evaluate the transmittance along the ray segment.
    if (sigma_s_scalar == 0)
    {
        result.interaction.set_invalid();

        // TODO implement

        //

        return result;
    }

    // TODO implement
    // result.interaction.incoming_distance = ...
    // result.transmittance_weight = ...

    //

    // Compute the position of the medium interaction if it is valid.
    if (result.interaction.is_valid())
    {
        result.interaction.position = ray.at(result.interaction.incoming_distance);
    }

    return result;
}
