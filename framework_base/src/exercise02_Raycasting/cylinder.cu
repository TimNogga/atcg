#include "cylinder.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>
#include "opg/raytracing/optixglm.h"


extern "C" __global__ void __intersection__cylinder()
{
    const glm::vec3 center = glm::vec3(0, 0, 0);
    const float     radius = 1;
    const float     half_height = 1.0f;
    const glm::vec3 axis   = glm::vec3(0, 0, 1);

    const glm::vec3 ray_orig = optixGetObjectRayOriginGLM();
    const glm::vec3 ray_dir  = optixGetObjectRayDirectionGLM();
    const float     ray_tmin = optixGetRayTmin();
    const float     ray_tmax = optixGetRayTmax();
    glm::vec3 O = ray_orig - center;
    glm::vec3 D = ray_dir;
    glm::vec2 O2 = glm::vec2(O.x, O.y);
    glm::vec2 D2 = glm::vec2(D.x, D.y);
    float p = glm::dot(D2, O2) / glm::dot(D2, D2); // p/2 actually
    float q = (glm::dot(O2, O2) - radius * radius) / glm::dot(D2, D2);

    float k = p*p - q;
    if (k < 0)
        return;

    // Try to report first interesction
    float t1 = -p - glm::sqrt(k);
    float hit_z1 = O.z + t1 * D.z;
    if ( t1 > ray_tmin && t1 < ray_tmax && glm::abs(hit_z1) <= half_height ) {
        if (optixReportIntersection( t1, 0 ))
            return;
    }

    // Report second intersection
    float t2 = -p + glm::sqrt(k);
    float hit_z2 = O.z + t2 * D.z;
    if ( t2 > ray_tmin && t2 < ray_tmax && glm::abs(hit_z2) <= half_height ) {
        if (optixReportIntersection( t2, 0 ))
            return;
    }

}

extern "C" __global__ void __closesthit__cylinder()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();
    


    // NOTE: optixGetObjectRayOrigin() and optixGetObjectRayDirection() are not available in closest hit programs.
    // const glm::vec3 object_ray_origin = optixGetObjectRayOriginGLM();
    // const glm::vec3 object_ray_dir    = optixGetObjectRayDirectionGLM();

    const glm::vec3 local_axis = glm::vec3(0, 0, 1);
    const float half_height = 1.0f;


    // Set incoming ray direction and distance
    si->incoming_ray_dir = world_ray_dir;
    si->incoming_distance = tmax;
    si->position = world_ray_origin + tmax * world_ray_dir;
    glm::vec3 local_position = optixTransformPointFromWorldToObjectSpace(si->position);
    const glm::vec3 local_up = glm::vec3(0, 0, 1);
    const glm::vec3 local_normal = glm::normalize(glm::vec3(local_position.x, local_position.y, 0));
     // Tangent corresponds to longitutde vector, orthogonal to "up" vector and normal
    const glm::vec3 local_tangent = glm::normalize(glm::cross(local_up, local_normal));

    // Transform local object space normal to world space normal
    si->normal = local_normal;
    si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
    si->normal = glm::normalize(si->normal);
    si->geom_normal = si->normal;

    // Transform local opbject space tangent to world space tangent
    si->tangent = local_tangent;
    si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
    si->tangent = glm::normalize(si->tangent);
    float v = (local_position.z + half_height) / (2 * half_height);
    float phi = glm::atan2(local_normal.y, local_normal.x);
    si->uv = glm::vec2(phi / glm::two_pi<float>(), v);



    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
}
