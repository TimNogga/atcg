#include "bsdfmodels.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/coordinates.h"

#include <optix.h>



// Schlick's approximation to the fresnel reflectance term
// See https://en.wikipedia.org/wiki/Schlick%27s_approximation
__forceinline__ __device__ float fresnel_schlick( const float F0, const float VdotH )
{
    return F0 + ( 1.0f - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}

__forceinline__ __device__ glm::vec3 fresnel_schlick( const glm::vec3 F0, const float VdotH )
{
    return F0 + ( glm::vec3(1.0f) - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}



__device__ glm::vec3 sampleHemisphereCosine(glm::vec2 uv)
{
    // Sample disk uniformly
    float r   = glm::sqrt(uv.x);
    float phi = 2.0f * M_PIf * uv.y;

    // Project disk sample onto hemisphere
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);
    float z = glm::sqrt( glm::max(0.0f, 1 - uv.x) );

    return glm::vec3(x, y, z);
}

__device__ float sampleHemisphereCosine_PDF(glm::vec3 result)
{
    return glm::max(0.0f, result.z) / M_PIf;
}



extern "C" __device__ BSDFEvalResult __direct_callable__diffuse_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const DiffuseBSDFData *sbt_data = *reinterpret_cast<const DiffuseBSDFData **>(optixGetSbtDataPointer());

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    glm::vec3 diffuse_bsdf_ndotl = clampedNdotL * sbt_data->diffuse_color / glm::pi<float>();
    float diffuse_pdf = clampedNdotL / M_PIf;

    BSDFEvalResult result;
    result.bsdf_value = diffuse_bsdf_ndotl;
    result.sampling_pdf = diffuse_pdf;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__diffuse_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng)
{
    const DiffuseBSDFData *sbt_data = *reinterpret_cast<const DiffuseBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0; // initialize with invalid sample

    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(si.normal, -si.incoming_ray_dir);
    if (NdotV <= 0)
    {
        return result;
    }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to the world coordinate system.
    glm::mat3 local_frame = opg::compute_local_frame(si.normal);

    glm::vec3 local_outgoing_ray_dir = sampleHemisphereCosine(rng.next2d());
    // transform local outgoing direction from tangent space to world space
    result.outgoing_ray_dir = local_frame * local_outgoing_ray_dir;

    // It is possible that light directions below the horizon are sampled..
    // If outgoing ray direction is below horizon, let the sampling fail!
    float NdotL = glm::dot(si.normal, result.outgoing_ray_dir);
    if (NdotL <= 0)
    {
        result.sampling_pdf = 0;
        return result;
    }

    glm::vec3 diffuse_bsdf_ndotl = NdotL * sbt_data->diffuse_color / M_PIf;
    float diffuse_pdf = NdotL / M_PIf;

    // Combined BRDF and sampling PDF
    result.bsdf_weight = sbt_data->diffuse_color; // NdotL/M_PIf cancels out
    result.sampling_pdf = diffuse_pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__diffuse_evalDiffuseAlbedo(const SurfaceInteraction &si)
{
    const DiffuseBSDFData *sbt_data = *reinterpret_cast<const DiffuseBSDFData **>(optixGetSbtDataPointer());
    return sbt_data->diffuse_color;
}


extern "C" __device__ BSDFEvalResult __direct_callable__specular_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    // No direct illumination on specular materials!
    // If the direction was not sampled from the specular BSDF itself, the sampling probability equals 0
    BSDFEvalResult result;
    result.bsdf_value = glm::vec3(0);
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__specular_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng)
{
    const SpecularBSDFData *sbt_data = *reinterpret_cast<const SpecularBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.outgoing_ray_dir = glm::reflect(si.incoming_ray_dir, si.normal);
    float clampedNdotL = glm::max(0.0f, glm::dot(result.outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));
    result.bsdf_weight = fresnel_schlick(sbt_data->specular_F0, clampedNdotL);
    result.sampling_pdf = 1;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__specular_evalDiffuseAlbedo(const SurfaceInteraction &si)
{
    // No diffuse component
    return glm::vec3(0);
}



extern "C" __device__ BSDFEvalResult __direct_callable__refractive_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    // No direct illumination on refractive materials!
    // If the direction was not sampled from the refractive BSDF itself, the sampling probability equals 0
    BSDFEvalResult result;
    result.bsdf_value = glm::vec3(0);
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__refractive_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng)
{
    const RefractiveBSDFData *sbt_data = *reinterpret_cast<const RefractiveBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0;

    bool outsidein = glm::dot(si.incoming_ray_dir, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta = outsidein ? 1.0f / sbt_data->index_of_refraction : sbt_data->index_of_refraction;

    glm::vec3 transmitted_ray_dir = glm::refract(si.incoming_ray_dir, interface_normal, eta);
    glm::vec3 reflected_ray_dir = glm::reflect(si.incoming_ray_dir, interface_normal);

    float F0 = (eta - 1) / (eta + 1);
    F0 = F0 * F0;

    float NdotL = glm::abs(glm::dot(si.incoming_ray_dir, interface_normal));

    float reflection_probability = fresnel_schlick(F0, NdotL);
    float transmission_probability = 1.0f - reflection_probability;

    if (glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability = 1.0f;
    }

    if (component_flags == +BSDFComponentFlag::IdealReflection && reflection_probability > 0)
    {
        result.bsdf_weight = glm::vec3(reflection_probability);
        result.outgoing_ray_dir = reflected_ray_dir;
        result.sampling_pdf = 1;
    }
    else if (component_flags == +BSDFComponentFlag::IdealTransmission && transmission_probability > 0)
    {
        result.bsdf_weight = glm::vec3(transmission_probability);
        result.outgoing_ray_dir = transmitted_ray_dir;
        result.sampling_pdf = 1;
    }
    else
    {
        // Stochastically select a reflection or transmission via russian roulette
        if (rng.next1d() < reflection_probability)
        {
            // Select the reflection event
            // We sample the BDSF exactly.
            result.bsdf_weight = glm::vec3(1);
            result.outgoing_ray_dir = reflected_ray_dir;
            result.sampling_pdf = reflection_probability;
        }
        else
        {
            // Select the transmission event
            // We sample the BDSF exactly.
            result.bsdf_weight = glm::vec3(1);
            result.outgoing_ray_dir = transmitted_ray_dir;
            result.sampling_pdf = transmission_probability;
        }
    }

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__refractive_evalDiffuseAlbedo(const SurfaceInteraction &si)
{
    // No diffuse component
    return glm::vec3(0);
}
