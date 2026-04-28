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

    /* Implement:
     * - Ray-cylinder intersection.
     * Hint: Use the function `bool optixReportIntersection(float hitT, unsigned int hitKind)` to report an intersection.
     * The hitKind can be set to 0 since we do not need it.
     */

    //
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


    /* Implement:
     * - Compute the position surface normal and tangent vector of the ray-cylinder intersection.
     * - Store these values in the SurfaceInteraction si.
     */

    // Default normal is (1,1,1) such that you can see something when implementing the intersection above.
    si->normal = glm::normalize(glm::vec3(1));

    //

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
}
