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


__device__ float henyey_greenstein_phase_function(float cos_theta, float g)
{
    float g2 = g*g;
    float area = 4*glm::pi<float>(); // area of sphere
    float phase = (1 - g2) / area * glm::pow((1 + g2 - 2*g*cos_theta), -1.5f);
    return phase;
}

__device__ glm::vec3 warp_square_to_sphere_henyey_greenstein(const glm::vec2 &uv, float g)
{
    float u1 = uv.x;
    float u2 = uv.y;

    float g2 = g*g;
    float d = (1 - g2) / (1 - g + 2*g*u1);
    float cos_theta = 0.5/g * (1 + g2 - d*d);

    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta*cos_theta));
    float phi = 2*glm::pi<float>() * u2;

    float x = sin_theta * glm::cos(phi);
    float y = sin_theta * glm::sin(phi);
    float z = cos_theta;

    return glm::vec3(x, y, z);
}

__device__ float warp_square_to_sphere_henyey_greenstein_pdf(const glm::vec3 & result, float g)
{
    return henyey_greenstein_phase_function(result.z, g);
}



extern "C" __device__ PhaseFunctionEvalResult __direct_callable__henyeygreenstein_evalPhaseFunction(const MediumInteraction &interaction, const glm::vec3 &outgoing_ray_dir)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    PhaseFunctionEvalResult result;
    // Since we can sample the phase function exactly, the sampling pdf is equal to the phase function itself.
    // The difference is that the phase function is in general allowed to return a "chromatic" value, and the sampling pdf returns a scalar value.
    result.sampling_pdf = henyey_greenstein_phase_function(glm::dot(interaction.incoming_ray_dir, outgoing_ray_dir), sbt_data->g);
    result.phase_function_value = glm::vec3(result.sampling_pdf);
    return result;
}

extern "C" __device__ PhaseFunctionSamplingResult __direct_callable__henyeygreenstein_samplePhaseFunction(const MediumInteraction &interaction, PCG32 &rng)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    glm::mat3 local_frame = opg::compute_local_frame(interaction.incoming_ray_dir);
    glm::vec3 local_outgoing_ray_dir = warp_square_to_sphere_henyey_greenstein(rng.next2d(), sbt_data->g);

    PhaseFunctionSamplingResult result;
    result.outgoing_ray_dir = local_frame * local_outgoing_ray_dir;
    result.sampling_pdf = warp_square_to_sphere_henyey_greenstein_pdf(local_outgoing_ray_dir, sbt_data->g);
    // result.phase_function_weight = glm::vec3(henyey_greenstein_phase_function(local_outgoing_ray_dir.z, sbt_data->g)) / result.sampling_pdf;
    result.phase_function_weight = glm::vec3(1);

    return result;
}
