#include "phasefunctions.cuh"
#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/coordinates.h"


__forceinline__ __device__ glm::vec3 warp_square_to_sphere_uniform(const glm::vec2 uv)
{
    float z   = uv.x * 2 - 1;
    float phi = uv.y * 2 * glm::pi<float>();

    float r = glm::sqrt( glm::max(0.0f, 1 - z*z) );
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_sphere_uniform_pdf(const glm::vec3 &dir)
{
    return 1 / (4 * glm::pi<float>());
}


// 


extern "C" __device__ PhaseFunctionEvalResult __direct_callable__henyeygreenstein_evalPhaseFunction(const MediumInteraction &interaction, const glm::vec3 &outgoing_ray_dir)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    /* Implement:
     * - Compute the phase-function value for `outgoing_ray_dir`.
     * - Compute the sampling probability of generating the `outgoing_ray_dir` using phase-function importance sampling.
     */

    PhaseFunctionEvalResult result;
    // Dummy implementation scatters uniformly into all directions.
    // result.sampling_pdf = 1/(4*glm::pi<float>());
    // result.phase_function_value = glm::vec3(1/(4*glm::pi<float>()));

    float cosTheta = glm::dot(-interaction.incoming_ray_dir, outgoing_ray_dir);
    float g = sbt_data->g;

    result.sampling_pdf = 1.0f / (4.0f * glm::pi<float>()) * (1.0f - g * g) / glm::pow(1.0f + g * g - 2.0f * g * cosTheta, 1.5f);
    result.phase_function_value = glm::vec3(result.sampling_pdf);

    //

    return result;
}

extern "C" __device__ PhaseFunctionSamplingResult __direct_callable__henyeygreenstein_samplePhaseFunction(const MediumInteraction &interaction, PCG32 &rng)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    PhaseFunctionSamplingResult result;
    // Dummy implementation samples the sphere uniformly.
    result.outgoing_ray_dir = warp_square_to_sphere_uniform(rng.next2d());
    result.sampling_pdf = warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir);
    // result.phase_function_weight = glm::vec3(warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir)) / warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir);
    result.phase_function_weight = glm::vec3(1);

    /* Implement:
     * - Sample a direction from the henyey greenstein phase function using the g-parameter stored in the sbt_data.
     * - Compute the respective phase function value.
     * - Compute the respective sampling probability.
     */

    float g = sbt_data->g;

    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();

    float cosTheta = 1.0f / (2.0f * g) * (1 + g * g - glm::pow((1.0f - g * g) / (1.0f - g + 2.0f * g * u1), 2.0f));
    float phi = u2 * glm::two_pi<float>();

    float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
    float x = sinTheta * glm::cos(phi);
    float y = sinTheta * glm::sin(phi);
    float z = cosTheta;

    glm::vec3 local_dir = glm::vec3(x, y, z);
    glm::mat3 local_frame = opg::compute_local_frame(-interaction.incoming_ray_dir);

    result.outgoing_ray_dir = local_frame * local_dir;
    result.sampling_pdf = 1.0f / (4.0f * glm::pi<float>()) * (1.0f - g * g) / glm::pow(1.0f + g * g - 2.0f * g * cosTheta, 1.5f);
    result.phase_function_weight = glm::vec3(1.0f); // = glm::vec3(result.sampling_pdf) / glm::vec3(result.sampling_pdf);
    //

    return result;
}
