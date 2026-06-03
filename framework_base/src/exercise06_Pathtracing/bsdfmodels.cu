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


__forceinline__ __device__ float D_GGX( const float NdotH, const float roughness )
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (M_PIf * d * d);
}

__forceinline__ __device__ float V_SmithJointGGX(float NdotL, float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    float lambdaV = NdotL * glm::sqrt(NdotV * NdotV * (1 - a2) + a2);
    float lambdaL = NdotV * glm::sqrt(NdotL * NdotL * (1 - a2) + a2);
    return 0.5f / (lambdaV + lambdaL);
}


//


extern "C" __device__ BSDFEvalResult __direct_callable__ggx_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const GGXBSDFData *sbt_data = *reinterpret_cast<const GGXBSDFData **>(optixGetSbtDataPointer());

    // Direction towards light
    glm::vec3 light_dir = outgoing_ray_dir;
    // Direction towards viewer
    glm::vec3 view_dir = -si.incoming_ray_dir;
    // Surface normal
    glm::vec3 normal = si.normal;

    // Don't apply BSDF if surface is viewed from below.
    // The BSDF is not importance sampled if surface is viewed from below.
    float NdotV = glm::dot(normal, view_dir);
    if (NdotV <= 0)
    {
        BSDFEvalResult result;
        result.bsdf_value = glm::vec3(0);
        result.sampling_pdf = 0;
        return result;
    }

    // The BSDF is 0 if the surface is illuminated from below.
    // If outgoing ray direction is below horizon, the sampling has failed.
    float NdotL = glm::dot(normal, light_dir);
    if (NdotL <= 0)
    {
        BSDFEvalResult result;
        result.bsdf_value = glm::vec3(0);
        result.sampling_pdf = 0;
        return result;
    }

    // Compute diffuse BSDF
    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / glm::pi<float>();

    // Compute specular BSDF
    glm::vec3 specular_bsdf = glm::vec3(0);
    // Only compute specular component if specular_f0 is not zero!
    if (glm::dot(sbt_data->specular_F0, sbt_data->specular_F0) > 1e-6)
    {
        glm::vec3 halfway = glm::normalize(light_dir + view_dir);
        float NdotH = glm::dot(halfway, normal);
        float LdotH = glm::dot(halfway, light_dir);

        // Normal distribution
        float D = D_GGX(NdotH, sbt_data->roughness);

        // Visibility
        float V = V_SmithJointGGX(NdotL, NdotV, sbt_data->roughness);

        // Fresnel
        glm::vec3 F = fresnel_schlick(sbt_data->specular_F0, LdotH);

        specular_bsdf = D * V * F;
    }

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * NdotL;
    result.sampling_pdf = 0; // Evaluation of bsdf importance sampling pdf is not used in this exercise.
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__ggx_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng)
{
    const GGXBSDFData *sbt_data = *reinterpret_cast<const GGXBSDFData **>(optixGetSbtDataPointer());

    // Direction towards viewer
    glm::vec3 view_dir = -si.incoming_ray_dir;
    // Surface normal
    glm::vec3 normal = si.normal;


    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample


    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(normal, view_dir);
    if (NdotV <= 0)
    {
        result.sampling_pdf = 0;
        return result;
    }

    /* Implement:
     * - Sample outgoing ray direction for diffuse lambert BRDF with probability <N, L> / PI
     * - Sample outgoing ray direction for specular GGX BRDF with probability D_GGX * <N, H> / (4 * <L, H>)
     */

    // Hint: reject light directions below the horizon

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to the world coordinate system.
    glm::mat3 local_frame = opg::compute_local_frame(normal);

    // Sample either the diffuse or specular lobe
    float diffuse_probability = glm::dot(sbt_data->diffuse_color, glm::vec3(1)) / (glm::dot(sbt_data->diffuse_color, glm::vec3(1)) + glm::dot(sbt_data->specular_F0, glm::vec3(1)));
    float specular_probability = 1 - diffuse_probability;
    // Make a random decision wether we sample a diffuse or glossy reflection
    if (rng.next1d() < diffuse_probability)
    {
        // The probability of entering this branch is `diffuse_probability`
        float branch_probability = diffuse_probability;

        // TODO implement diffuse reflection

        float u1 = rng.next1d();
        float u2 = rng.next1d();

        float r = sqrt(u1);
        float phi = 2.0f * M_PI * u2;

        float x = r * cos(phi);
        float y = r * sin(phi);
        float z = sqrt(1 - u1);

        glm::vec3 local_dir(x, y, z);

        glm::vec3 L = local_frame * local_dir;

        float NdotL = glm::dot(normal, L);

        if (NdotL <= 0.0f) {
            result.sampling_pdf = 0.0f;
            return result;
        }

        result.outgoing_ray_dir = L;
        result.bsdf_weight = sbt_data->diffuse_color;
        result.sampling_pdf = branch_probability * NdotL / M_PI;

        //
    }
    else
    {
        // The probability of entering this branch is 1 - the probability of the above event
        float branch_probability = specular_probability;

        // TODO implement specular reflection

        float u1 = rng.next1d();
        float u2 = rng.next1d();

        float NdotH = sqrt((1.0f - u1) / (1.0f + (sbt_data->roughness * sbt_data->roughness) * u1));
        float sinThetaH = sqrt(glm::max(0.0f, 1.0f - NdotH * NdotH));

        float phi = 2.0f * M_PI * u2;

        float H_local_x = sinThetaH * cos(phi);
        float H_local_y = sinThetaH * sin(phi);
        float H_local_z = NdotH;

        glm::vec3 H_local(H_local_x, H_local_y, H_local_z);
        glm::vec3 H = local_frame * H_local;

        glm::vec3 L = 2.0f * glm::dot(view_dir, H) * H - view_dir;

        float NdotL = glm::dot(normal, L);

        if (NdotL <= 0.0f) {
            result.sampling_pdf = 0.0f;
            return result;
        }

        result.outgoing_ray_dir = L;
        result.bsdf_weight = sbt_data->specular_F0;
        result.sampling_pdf = branch_probability * D_GGX(NdotH, sbt_data->roughness) * NdotH / (4 * glm::dot(L, H));

        //
    }

    return result;
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


    // Determine surface parameters
    bool outsidein = glm::dot(si.incoming_ray_dir, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta = outsidein ? 1.0f / sbt_data->index_of_refraction : sbt_data->index_of_refraction;

    // Compute outgoing ray directions
    glm::vec3 transmitted_ray_dir = glm::refract(si.incoming_ray_dir, interface_normal, eta);
    glm::vec3 reflected_ray_dir = glm::reflect(si.incoming_ray_dir, interface_normal);

    // Fresnel reflectance at normal incidence
    float F0 = (eta - 1) / (eta + 1);
    F0 = F0 * F0;

    float NdotL = glm::abs(glm::dot(si.incoming_ray_dir, interface_normal));

    // Reflection an transmission probabilities
    float reflection_probability = fresnel_schlick(F0, NdotL);
    float transmission_probability = 1.0f - reflection_probability;
    if (glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability = 1.0f;
    }


    // Compute sampling result
    BSDFSamplingResult result;
    result.sampling_pdf = 0;

    if (component_flags == +BSDFComponentFlag::IdealReflection && reflection_probability > 0)
    {
        // The caller explicitly requested to sample a reflection
        result.bsdf_weight = glm::vec3(reflection_probability);
        result.outgoing_ray_dir = reflected_ray_dir;
        result.sampling_pdf = 1;
    }
    else if (component_flags == +BSDFComponentFlag::IdealTransmission && transmission_probability > 0)
    {
        // The caller explicitly requested to sample a transmission
        result.bsdf_weight = glm::vec3(transmission_probability);
        result.outgoing_ray_dir = transmitted_ray_dir;
        result.sampling_pdf = 1;
    }
    else
    {
        /* Implement:
         * - Sample outgoing ray direction proportional to the reflection/transmission probability
         */

        // TODO implement

        

        //
    }

    return result;
}
