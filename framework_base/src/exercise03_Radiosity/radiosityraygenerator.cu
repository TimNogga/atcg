#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "radiosityraygenerator.cuh"

#include "opg/hostdevice/color.h"
#include "opg/memory/stack.h"
#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"

__constant__ RadiosityLaunchParams params;

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







//
// Helper functions for the handling of primitive data
//

// The data associated with a triangle primitive for the computation of the transport matrix K
struct PrimitiveData
{
    glm::vec3 position[3];  // Vertex positions
    glm::vec3 normal[3];    // Vertex normals
    uint32_t  matrix_index; // The row and column index in the transport matrix associated with this primitive
    glm::vec3 albedo;       // The albedo of this primitive
    float area;             // The area of this primitive
};

__device__ PrimitiveData makePrimitiveData(const ComputeFormFactorMatrixInstanceData &instance_data, uint32_t prim_idx)
{
    PrimitiveData primitive_data;

    primitive_data.matrix_index = instance_data.form_factor_matrix_offset + prim_idx;

    glm::uvec3 vertex_indices;
    if (instance_data.indices.elmt_byte_size == sizeof(glm::u16vec3))
    {
        auto indices16 = instance_data.indices.asType<glm::u16vec3>();
        vertex_indices = indices16[prim_idx];
    }
    else
    {
        auto indices32 = instance_data.indices.asType<glm::u32vec3>();
        vertex_indices = indices32[prim_idx];
    }


    for (uint32_t tri_idx = 0; tri_idx < 3; ++tri_idx)
    {
        uint32_t vert_idx = vertex_indices[tri_idx];

        primitive_data.position[tri_idx] = glm::vec3(instance_data.transform * glm::vec4(instance_data.positions[vert_idx], 1.0f));
        primitive_data.normal[tri_idx] = glm::normalize(glm::vec3(instance_data.transform * glm::vec4(instance_data.normals[vert_idx], 0.0f)));
    }

//

    return primitive_data;
}

//


extern "C" __global__ void __raygen__generateRadiosity()
{
    const uint3 idx = optixGetLaunchIndex();

    bool same_geometry = (params.instance_1.form_factor_matrix_offset == params.instance_2.form_factor_matrix_offset);
    if (same_geometry && idx.x > idx.y) return;

    PrimitiveData primitive_1 = makePrimitiveData(params.instance_1, idx.x);
    PrimitiveData primitive_2 = makePrimitiveData(params.instance_2, idx.y);
    
    int i = primitive_1.matrix_index;
    int j = primitive_2.matrix_index;
    int matrix_size = params.form_factor_matrix_size;

    glm::vec3 center_i = (primitive_1.position[0] + primitive_1.position[1] + primitive_1.position[2]) / 3.0f;
    glm::vec3 center_j = (primitive_2.position[0] + primitive_2.position[1] + primitive_2.position[2]) / 3.0f;
    
    glm::vec3 n_i = primitive_1.normal[0]; 
    glm::vec3 n_j = primitive_2.normal[0];

    glm::vec3 e1_i = primitive_1.position[1] - primitive_1.position[0];
    glm::vec3 e2_i = primitive_1.position[2] - primitive_1.position[0];
    float area_i = 0.5f * glm::length(glm::cross(e1_i, e2_i));

    glm::vec3 e1_j = primitive_2.position[1] - primitive_2.position[0];
    glm::vec3 e2_j = primitive_2.position[2] - primitive_2.position[0];
    float area_j = 0.5f * glm::length(glm::cross(e1_j, e2_j));

    glm::vec3 dir = center_j - center_i;
    float dist = glm::length(dir);
    
    float F_ij = 0.0f;

    if (dist >= 1e-5f) 
    {
        dir /= dist;

        float cos_theta_i = glm::dot(n_i, dir);
        float cos_theta_j = glm::dot(n_j, -dir);

        if (cos_theta_i > 0.0f && cos_theta_j > 0.0f)
        {
            uint32_t is_occluded = 1; 
           
            optixTrace(
                params.traversable_handle,
                make_float3(center_i.x, center_i.y, center_i.z),
                make_float3(dir.x, dir.y, dir.z),
                params.scene_epsilon, dist - params.scene_epsilon, 0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                0, 0, 1, 
                is_occluded
            );
     
            if (is_occluded == 0) 
            {
                F_ij = (1.0f / M_PI) * ((cos_theta_i * cos_theta_j) / (dist * dist)) * area_j;
                
                if (F_ij > 1.0f) F_ij = 1.0f; 
            }
        }
    }

    params.form_factor_matrix[i * matrix_size + j] = F_ij;

 
    if (area_j > 0.0f) {
        float F_ji = F_ij * (area_i / area_j);
        if (F_ji > 1.0f) F_ji = 1.0f;
        params.form_factor_matrix[j * matrix_size + i] = F_ji; // We must explicitly initialize every cell in the form factor matrix to prevent uninitialized memory from causing NaN  during the Jacobi solver phase i guess

    }
}
extern "C" __global__ void __raygen__renderRadiosity()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM();
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();

    // Index of current pixel
    const glm::uvec2 pixel_index = glm::uvec2(launch_idx.x, launch_idx.y);

    glm::vec2 uv = (glm::vec2(pixel_index)+0.5f) / glm::vec2(params.image_width, params.image_height);
    uv = 2.0f*uv - 1.0f; // [0, 1] -> [-1, 1]

    glm::vec3 ray_origin;
    glm::vec3 ray_direction;
    spawn_camera_ray(params.camera, uv, ray_origin, ray_direction);


    SurfaceInteraction si;
    traceWithDataPointer<SurfaceInteraction>(
            params.traversable_handle,
            ray_origin,
            ray_direction,
            params.scene_epsilon,                   // tmin: Start ray at ray_origin + tmin * ray_direction
            std::numeric_limits<float>::infinity(), // tmax: End ray at ray_origin + tmax * ray_direction
            params.surface_interaction_trace_params,
            &si
    );

    glm::vec3 radiosity_output = glm::vec3(0);

    // If we have a valid surface interaction with an emitter...
    if (si.is_finite() && si.emitter != nullptr)
    {
        radiosity_output = si.emitter->evalLight(si);
    }

    // Write linear output color
    params.output_radiance(pixel_index).value() = radiosity_output;
}
