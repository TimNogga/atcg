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



__forceinline__ __device__ glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2 &uv)
{
    // Sample disk uniformly
    float r   = glm::sqrt(uv.x);
    float phi = 2.0f*glm::pi<float>() * uv.y;

    // Project disk sample onto hemisphere
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);
    float z = glm::sqrt( glm::max(0.0f, 1 - uv.x) );

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_hemisphere_cosine_pdf(const glm::vec3 &result)
{
    return glm::max(0.0f, result.z) / glm::pi<float>();
}

__forceinline__ __device__ glm::vec3 warp_square_to_hemisphere_ggx(const glm::vec2 &uv, float roughness)
{
    // GGX NDF sampling
    float cos_theta = glm::sqrt((1.0f - uv.x) / (1.0f + (roughness*roughness - 1.0f) * uv.x));
    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta*cos_theta));
    float phi      = 2.0f*glm::pi<float>() * uv.y;

    float x = sin_theta * glm::cos(phi);
    float y = sin_theta * glm::sin(phi);
    float z = cos_theta;

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_hemisphere_ggx_pdf(const glm::vec3 &result, float roughness)
{
    return D_GGX(result.z, roughness) * glm::max(0.0f, result.z);
}

__forceinline__ __device__ float warp_normal_to_reflected_direction_pdf(const glm::vec3 &reflected_dir, const glm::vec3 &normal)
{
    return 1 / glm::abs(4*glm::dot(reflected_dir, normal));
}



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


    // Sample the sum of diffuse and specular lobes
    // The probability that the diffuse BSDF was used to generate a ray
    float diffuse_probability = glm::dot(sbt_data->diffuse_color, glm::vec3(1)) / (glm::dot(sbt_data->diffuse_color, glm::vec3(1)) + glm::dot(sbt_data->specular_F0, glm::vec3(1)));
    // The probability that the specular BSDF was used to generate a ray
    float specular_probability = 1 - diffuse_probability;


    // Compute diffuse BSDF
    glm::vec3 diffuse_bsdf_ndotl = NdotL * sbt_data->diffuse_color / glm::pi<float>();
    // Probability of sampling the outgoing_ray_dir given that the diffuse BSDF was sampled
    float diffuse_pdf = NdotL / glm::pi<float>();


    // Compute specular BSDF
    glm::vec3 specular_bsdf_ndotl = glm::vec3(0);
    // Probability of sampling the outgoing_ray_dir given that the specular BSDF was sampled
    float specular_pdf = 0;
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

        // Specular BSDF
        specular_bsdf_ndotl = NdotL * D * V * F;

        // Specular sampling PDF
        float halfway_pdf = D_GGX(NdotH, sbt_data->roughness) * NdotH;
        float halfway_to_outgoing_pdf = warp_normal_to_reflected_direction_pdf(outgoing_ray_dir, halfway); // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    BSDFEvalResult result;
    result.bsdf_value = diffuse_bsdf_ndotl + specular_bsdf_ndotl;
    result.sampling_pdf = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
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
    result.sampling_pdf = 1; // initialize with the probability of sampling this BSDF given that we are executing this code.

    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(normal, view_dir);
    if (NdotV <= 0)
    {
        result.sampling_pdf = 0;
        return result;
    }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to the world coordinate system.
    glm::mat3 local_frame = opg::compute_local_frame(normal);

    float diffuse_probability = glm::dot(sbt_data->diffuse_color, glm::vec3(1)) / (glm::dot(sbt_data->diffuse_color, glm::vec3(1)) + glm::dot(sbt_data->specular_F0, glm::vec3(1)));
    float specular_probability = 1 - diffuse_probability;
    // Make a random decision wether we sample a diffuse or glossy reflection
    if (rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        glm::vec3 local_outgoing_ray_dir = warp_square_to_hemisphere_cosine(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.outgoing_ray_dir = local_frame * local_outgoing_ray_dir;
    }
    else
    {
        // Sample light direction from specular bsdf
        glm::vec3 local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), sbt_data->roughness);
        // Transform local halfway vector from tangent space to world space
        glm::vec3 halfway = local_frame * local_halfway;
        result.outgoing_ray_dir = glm::reflect(si.incoming_ray_dir, halfway);
    }

    // It is possible that light directions below the horizon are sampled..
    // If outgoing ray direction is below horizon, let the sampling fail!
    float NdotL = glm::dot(normal, result.outgoing_ray_dir);
    if (NdotL <= 0)
    {
        result.sampling_pdf = 0;
        return result;
    }

    glm::vec3 diffuse_bsdf = NdotL * sbt_data->diffuse_color / glm::pi<float>();
    float diffuse_pdf = NdotL / glm::pi<float>();

    glm::vec3 specular_bsdf = glm::vec3(0);
    float specular_pdf = 0;
    // Only compute specular component if specular_f0 is not zero!
    if (glm::dot(sbt_data->specular_F0, sbt_data->specular_F0) > 1e-6)
    {
        glm::vec3 halfway = glm::normalize(result.outgoing_ray_dir + view_dir);
        float NdotH = glm::dot(halfway, normal);
        float HdotV = glm::dot(halfway, result.outgoing_ray_dir);

        // Normal distribution
        float D = D_GGX(NdotH, sbt_data->roughness);

        // Visibility
        float V = V_SmithJointGGX(NdotL, NdotV, sbt_data->roughness);

        // Fresnel
        glm::vec3 F = fresnel_schlick(sbt_data->specular_F0, HdotV);

        specular_bsdf = NdotL * D * V * F;

        float halfway_pdf = D_GGX(NdotH, sbt_data->roughness) * NdotH;
        float halfway_to_outgoing_pdf = warp_normal_to_reflected_direction_pdf(result.outgoing_ray_dir, halfway); // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    // Combined BRDF and sampling PDF
    result.bsdf_weight = (diffuse_bsdf + specular_bsdf) / (diffuse_probability * diffuse_pdf + specular_probability * specular_pdf);
    result.sampling_pdf = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;

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



extern "C" __device__ BSDFEvalResult __direct_callable__nullbsdf_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    // No direct illumination on null bsdf!
    // If the direction was not sampled from the nullbsdf BSDF itself, the sampling probability equals 0
    BSDFEvalResult result;
    result.bsdf_value = glm::vec3(0);
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__nullbsdf_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng)
{
    // Incomming light direction is *not* bent or attenuated by null bsdf.

    // Compute sampling result
    BSDFSamplingResult result;
    result.outgoing_ray_dir = si.incoming_ray_dir;
    result.sampling_pdf = 1;
    result.bsdf_weight = glm::vec3(1);
    return result;
}
