#include "mesh.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>
#include "opg/raytracing/optixglm.h"


extern "C" __global__ void __closesthit__mesh()
{
    SurfaceInteraction*                 si       = getPayloadDataPointer<SurfaceInteraction>();
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());
    const MeshShapeData*                mesh_data = reinterpret_cast<const MeshShapeData*>(sbt_data->shape);

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();


    // NOTE: optixGetObjectRayOrigin() and optixGetObjectRayDirection() are not available in closest hit programs.
    // const glm::vec3 object_ray_origin = optixGetObjectRayOriginGLM();
    // const glm::vec3 object_ray_dir    = optixGetObjectRayDirectionGLM();


    // Set incoming ray direction and distance
    si->incoming_ray_dir = world_ray_dir;
    si->incoming_distance = tmax;


    const uint32_t  prim_idx = optixGetPrimitiveIndex();
    const glm::vec2 barys    = optixGetTriangleBarycentricsGLM();

    // Indices of triangle vertices in the mesh
    glm::uvec3 tri = glm::uvec3(0u);
    if (mesh_data->indices.elmt_byte_size == sizeof(glm::u32vec3))
    {
        // Indices stored as 32-bit unsigned integers
        const glm::u32vec3* indices = reinterpret_cast<glm::u32vec3*>(mesh_data->indices.data);
        tri = glm::uvec3(indices[prim_idx]);
    }
    else
    {
        // Indices stored as 16-bit unsigned integers
        const glm::u16vec3* indices = reinterpret_cast<glm::u16vec3*>(mesh_data->indices.data);
        tri = glm::uvec3(indices[prim_idx]);
    }

    // Compute local position
    const glm::vec3 P0 = mesh_data->positions[tri.x];
    const glm::vec3 P1 = mesh_data->positions[tri.y];
    const glm::vec3 P2 = mesh_data->positions[tri.z];
    si->position = (1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
    // Transform local position to world position
    si->position = optixTransformPointFromObjectToWorldSpace(si->position);

    si->geom_normal = glm::cross(P1-P0, P2-P0);
    si->geom_normal = optixTransformNormalFromObjectToWorldSpace(si->geom_normal);
    si->geom_normal = glm::normalize(si->geom_normal);

    if (mesh_data->uvs)
    {
        const glm::vec2 UV0 = mesh_data->uvs[tri.x];
        const glm::vec2 UV1 = mesh_data->uvs[tri.y];
        const glm::vec2 UV2 = mesh_data->uvs[tri.z];
        si->uv = (1.0f-barys.x-barys.y)*UV0 + barys.x*UV1 + barys.y*UV2;
    }
    else
    {
        // Assume:
        // vertex0.uv = (0, 0)
        // vertex1.uv = (1, 0)
        // vertex2.uv = (0, 1)
        si->uv = barys;
    }

    if (mesh_data->normals)
    {
        const glm::vec3 N0 = mesh_data->normals[tri.x];
        const glm::vec3 N1 = mesh_data->normals[tri.y];
        const glm::vec3 N2 = mesh_data->normals[tri.z];
        si->normal = (1.0f-barys.x-barys.y)*N0 + barys.x*N1 + barys.y*N2;
        si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
        si->normal = glm::normalize(si->normal);
    }
    else
    {
        si->normal = si->geom_normal;
    }

    if (mesh_data->tangents)
    {
        const glm::vec3 T0 = mesh_data->tangents[tri.x];
        const glm::vec3 T1 = mesh_data->tangents[tri.y];
        const glm::vec3 T2 = mesh_data->tangents[tri.z];
        si->tangent = ( 1.0f-barys.x-barys.y)*T0 + barys.x*T1 + barys.y*T2;
        si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
        si->tangent = glm::normalize(si->tangent - glm::dot(si->tangent, si->normal) * si->normal );
    }
    else
    {
        const glm::vec2 UV0 = mesh_data->uvs[tri.x];
        const glm::vec2 UV1 = mesh_data->uvs[tri.y];
        const glm::vec2 UV2 = mesh_data->uvs[tri.z];
        glm::vec3 E1 = P1 - P0;
        glm::vec3 E2 = P2 - P0;
        glm::vec2 dUV1 = UV1 - UV0;
        glm::vec2 dUV2 = UV2 - UV0;

 
        const float inv_det = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
        glm::vec3 obj_tangent = inv_det * (dUV2.y * E1 - dUV1.y * E2);

        si->tangent = optixTransformVectorFromObjectToWorldSpace(obj_tangent);
        si->tangent = glm::normalize(si->tangent - glm::dot(si->tangent, si->normal) * si->normal);

    }

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
}

extern "C" __global__ void __anyhit__mesh()
{
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());
    const MeshShapeData*                mesh_data = reinterpret_cast<const MeshShapeData*>(sbt_data->shape);

    const uint32_t prim_idx = optixGetPrimitiveIndex();
    const float2   barys    = optixGetTriangleBarycentrics();


    // Only perform anyhit check if uv coordinates are present
    if (!mesh_data->uvs)
        return;


    // Indices of triangle vertices in the mesh
    glm::uvec3 tri = glm::uvec3(0u);
    if (mesh_data->indices.elmt_byte_size == sizeof(glm::u32vec3))
    {
        // Indices stored as 32-bit unsigned integers
        const glm::u32vec3* indices = reinterpret_cast<glm::u32vec3*>(mesh_data->indices.data);
        tri = glm::uvec3(indices[prim_idx]);
    }
    else
    {
        // Indices stored as 16-bit unsigned integers
        const glm::u16vec3* indices = reinterpret_cast<glm::u16vec3*>(mesh_data->indices.data);
        tri = glm::uvec3(indices[prim_idx]);
    }


    const glm::vec2 UV0 = mesh_data->uvs[tri.x];
    const glm::vec2 UV1 = mesh_data->uvs[tri.y];
    const glm::vec2 UV2 = mesh_data->uvs[tri.z];
    glm::vec2 uv = (1.0f-barys.x-barys.y)*UV0 + barys.x*UV1 + barys.y*UV2;

  
    glm::vec2 s = uv;
    const int level = 3;
    for (int i = 0; i < level; ++i)
    {
        s *= 3.0f;
        glm::vec2 idx = glm::floor(s);
        int ix = static_cast<int>(idx.x);
        int iy = static_cast<int>(idx.y);
        if ((ix % 3 == 1) && (iy % 3 == 1))
        {
            optixIgnoreIntersection();
            return;
        }
        s = s - idx;
    }
}
