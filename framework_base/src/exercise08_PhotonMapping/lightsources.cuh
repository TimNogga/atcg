#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"

struct PointLightData
{
    glm::vec3 position;
    glm::vec3 intensity; // W/sr at light source
};

struct DirectionalLightData
{
    glm::vec3 direction;
    glm::vec3 irradiance_at_receiver; // W/m^2 at receiver, dA orthogonal to direction!
};

struct SphereLightData
{
    glm::vec3 position;
    float     radius;
    glm::vec3 radiance; // W/(m^2*sr) at light source
};

struct MeshLightData
{
    // The probability of each triangle is proportional to its area.
    // This is the cummulative distribution for each triangle in the mesh.
    opg::BufferView<float> mesh_cdf;

    // The index buffer of the mesh
    opg::GenericBufferView mesh_indices;
    // The vertex-position buffer of the mesh
    opg::BufferView<glm::vec3> mesh_positions;

    // The transformation from the local coordinate system to the world coordinate system of
    // the instance that this emitter is attached to.
    glm::mat4 local_to_world;
    glm::mat4 world_to_local;

    // The radiance that is uniformly emitted at every surface point of the mesh.
    glm::vec3 radiance;

    // Does the mesh emit light on the front and back side, or only on the front side?
    bool double_sided;

    // The total surface area of the mesh
    float total_surface_area;
};
