#pragma once

#include "opg/glmwrapper.h"

enum class KDNodeType : int
{
    Empty,
    Leaf,       // Leaf with a single photon.
    DoubleLeaf, // Leaf with two photons stored next to each other.
    AxisX,      // Inner node with single photon and two child nodes.
    AxisY,      // Inner node with single photon and two child nodes.
    AxisZ       // Inner node with single photon and two child nodes.
};

// The data that is stored in the photon map
struct PhotonData
{
    // Recorded position of the photon
    glm::vec3 position;
    // Recorded surface normal of the photon
    glm::vec3 normal;
    // The resulting *irradiance* [W/m^2] of the photon at the recorded surface, i.e. radiance * <N,L>
    glm::vec3 irradiance_weight;
    // The type of this node in the KD-tree, indicating how to continue with the traversal of the tree.
    KDNodeType node_type;
};

// The data that is stored per-pixel encoding photon gather requests
struct PhotonGatherData
{
    // The gathering request filled in the normal path tracing pipeline:
    // The surface position of the gathering request
    glm::vec3 position;
    // The surface normal of the gathering request
    glm::vec3 normal;
    // The contribution of this gathering request to the final pixel color, this already includes albedo/pi of BSDF at the surface.
    glm::vec3 throughput; // Unit: [1/sr] due to inclusion of BSDF.
    // Additional data for progressive photon mapping:
    // The adjusted gathering radius
    float       gather_radius_sq;
    // Number of photons accumulated so far
    float       photon_count;
    // Power (or integrated irradiance) estimate inside of gather region
    glm::vec3   total_power;
};

struct PhotonMapStoreCount
{
    // The number of photons that are actually stored in the photon map.
    uint32_t actual_count;
    // The number of photons that **should** have been stored in the photon map if it had unlimited capacity.
    uint32_t desired_count;
};
