#include "bsdfmodels.cuh"

#include "opg/scene/utility/interaction.cuh"

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


extern "C" __device__ BSDFEvalResult __direct_callable__opaque_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());
    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf * glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = diffuse_bsdf;
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__opaque_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    if (!has_flag(component_flags, BSDFComponentFlag::IdealReflection))
        return result;
    if (glm::dot(sbt_data->specular_F0, sbt_data->specular_F0) < 1e-6)
        return result;

    result.outgoing_ray_dir = glm::reflect(si.incoming_ray_dir, si.normal);
    result.bsdf_weight = sbt_data->specular_F0; // TODO evaluate Schlick's Fresnel!
    result.sampling_pdf = 1;

    return result;
}


extern "C" __device__ BSDFEvalResult __direct_callable__refractive_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    // No direct illumination on refractive materials!
    BSDFEvalResult result;
    result.bsdf_value = glm::vec3(0);
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__refractive_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
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

    return result;
}



// 



//
// Phong BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__phong_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const PhongBSDFData *sbt_data = *reinterpret_cast<const PhongBSDFData **>(optixGetSbtDataPointer());

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;
    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * Phong BRDF
     */

    // TODO implement
    glm::vec3 reflection_dir = glm::reflect(si.incoming_ray_dir, si.normal);
    float m = sbt_data->exponent;
    specular_bsdf = sbt_data->specular_F0 * (m + 2.0f) / (2.0f * M_PIf) * glm::pow(glm::max(glm::dot(reflection_dir, outgoing_ray_dir), 0.0f), m);

    //

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}


//
// Ward BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__ward_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const WardBSDFData *sbt_data = *reinterpret_cast<const WardBSDFData **>(optixGetSbtDataPointer());

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;
    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * - Anisotropic Geisler-Moroder variant of the Ward BRDF
     *     - Beckmann normal distribution
     *     - Geisler-Moroder normalization
     *     - Schlick's Fresnel approximation
     */

    // TODO implement

    glm::vec3 halfway = -si.incoming_ray_dir + outgoing_ray_dir;
    float alpha = sbt_data->roughness_tangent, beta = sbt_data->roughness_bitangent;
    const glm::vec3& tangent = si.tangent;
    glm::vec3 bitangent = glm::normalize(glm::cross(si.normal, tangent));

    float HdotN = glm::max(glm::dot(halfway, si.normal), 1e-6f);
    float HdotH = glm::dot(halfway, halfway);
    float HdotX = glm::dot(halfway, tangent);
    float HdotY = glm::dot(halfway, bitangent);

    float f = glm::exp(-((HdotX * HdotX) / (alpha * alpha) + (HdotY * HdotY) / (beta * beta)) / (HdotN * HdotN)) * HdotH / (glm::pow(HdotN, 4) * M_PIf * alpha * beta);
    glm::vec3 fresnel = fresnel_schlick(sbt_data->specular_F0, glm::dot(outgoing_ray_dir, glm::normalize(halfway)));
    specular_bsdf = f * fresnel;

    //

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}


//
// GGX BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__ggx_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const GGXBSDFData *sbt_data = *reinterpret_cast<const GGXBSDFData **>(optixGetSbtDataPointer());

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;

    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * - Anisotropic microfacet BRDF with
     *     - GGX microfacet distribution
     *     - Smith geometric masking/shadowing term
     *     - Schlick's Fresnel approximation
     */

    // TODO implement

    const glm::vec3& incoming_ray_dir = -si.incoming_ray_dir;

    glm::vec3 halfway = glm::normalize(incoming_ray_dir + outgoing_ray_dir);
    float alpha = sbt_data->roughness_tangent, beta = sbt_data->roughness_bitangent;

    const glm::vec3& tangent = si.tangent;
    glm::vec3 bitangent = glm::normalize(glm::cross(si.normal, si.tangent));

    float HdotN = glm::max(glm::dot(halfway, si.normal), 1e-6f);
    float HdotX = glm::dot(halfway, tangent);
    float HdotY = glm::dot(halfway, bitangent);
    float VdotH = glm::dot(outgoing_ray_dir, halfway);
    float NdotL = glm::dot(si.normal, incoming_ray_dir);
    float NdotV = glm::dot(si.normal, outgoing_ray_dir);

    float tan_term_incoming = (alpha * alpha * glm::pow(glm::dot(tangent, incoming_ray_dir), 2) + beta * beta * glm::pow(glm::dot(bitangent, incoming_ray_dir), 2)) / glm::pow(glm::dot(si.normal, incoming_ray_dir), 2);
    float tan_term_outgoing = (alpha * alpha * glm::pow(glm::dot(tangent, outgoing_ray_dir), 2) + beta * beta * glm::pow(glm::dot(bitangent, outgoing_ray_dir), 2)) / glm::pow(glm::dot(si.normal, outgoing_ray_dir), 2);

    float denom = HdotN * HdotN * (1.0f + (1.0f / (HdotN * HdotN)) * ((HdotX * HdotX) / (alpha * alpha) + (HdotY * HdotY) / (beta * beta)));
    float dist = 1.0f / (M_PIf * alpha * beta * denom * denom);
    float masking = 1.0f / (1.0f + ((-1.0f + glm::sqrt(1 + tan_term_incoming)) / 2.0f) + ((-1.0f + glm::sqrt(1 + tan_term_outgoing)) / 2.0f));
    glm::vec3 fresnel = fresnel_schlick(sbt_data->specular_F0, VdotH);

    specular_bsdf = dist * masking * fresnel / (4.0f * NdotL * NdotV);

    //


    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}



// Shared dummy BSDF sampling method
extern "C" __device__ BSDFSamplingResult __direct_callable__phong_ward_ggx_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    // Importance sampling of glossy BSDFs is added in a future exercise...
    // For now, there is no importance sampling support for this BSDF
    return result;
}
