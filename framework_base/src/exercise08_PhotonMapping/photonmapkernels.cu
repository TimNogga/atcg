#include "photonmapkernels.h"

#include "opg/hostdevice/misc.h"
#include "opg/hostdevice/color.h"
#include "opg/memory/stack.h"
#include "opg/kernels/launch.h"

#include <algorithm>

#pragma cuda_source_property_format=OBJ

//
// Photon gathering
//
__global__ void resetPhotonGatherDataKernel(opg::BufferView<PhotonGatherData> gather_data, float gather_radius_sq)
{
    const uint32_t gather_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (gather_index >= gather_data.count)
        return;

    // Reset "internal" state
    gather_data[gather_index].gather_radius_sq = gather_radius_sq;
    gather_data[gather_index].photon_count = 0;
    gather_data[gather_index].total_power = glm::vec3(0.0f);
}

void resetPhotonGatherData(opg::BufferView<PhotonGatherData> gather_data, float gather_radius_sq)
{
    const int blockSize  = 512; // 512 is a size that works well with modern GPUs.
    const int blockCount = ceil_div<int>(gather_data.count, blockSize); // Spawn enough blocks such that each pair of elements is added in in its own thread.

    // Launch the kernel on the GPU!
    resetPhotonGatherDataKernel<<<blockCount, blockSize>>>(gather_data, gather_radius_sq);
}

__device__ __forceinline__ void accumulatePhoton( const PhotonData &photon,
                        const glm::vec3 &gather_position,
                        const glm::vec3 &gather_normal,
                        const glm::vec3 &gather_throughput,
                        const float &gather_radius_sq,
                        uint32_t& acc_photon_count, glm::vec3& acc_weight )
{
    glm::vec3 dist = gather_position - photon.position;
    if (glm::dot(dist, dist) < gather_radius_sq)
    {
        float cos_theta = glm::dot(photon.normal, gather_normal);

        if ( cos_theta > 0.8f ) // TODO threshold on curved surfaces!
        {
            glm::vec3 weight = photon.irradiance_weight * gather_throughput;
            acc_photon_count++;
            acc_weight += weight;
        }
    }
}

#define MAX_DEPTH 20 // one MILLION photons
__global__ void gatherPhotonsKernel(
    opg::BufferView<PhotonData> photon_map, // contains position, normal, irradiance_weight and node_type
    opg::BufferView<PhotonGatherData> gather_data, // contains gather requests: position, normal, throughput, initial radius, etc.
    opg::BufferView<glm::vec3> output_radiance, // store computed radiance per request
    PhotonMapStoreCount *photon_map_store_count_ptr, // total number of photons in photon map
    uint32_t *total_emitted_photon_count_ptr, // total number of photons emitted
    float alpha // blending parameter for progressive photon mapping
    )
{
    
    const uint32_t gather_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (gather_index >= gather_data.count)
        return;

    glm::vec3 gather_position   = gather_data[gather_index].position;
    glm::vec3 gather_normal     = gather_data[gather_index].normal;
    glm::vec3 gather_throughput = gather_data[gather_index].throughput;

    float  gather_radius_sq    = gather_data[gather_index].gather_radius_sq;
    float  gather_photon_count = gather_data[gather_index].photon_count;

    glm::vec3 gather_total_power = gather_data[gather_index].total_power;

    /* Implement:
     * - Traverse the photon map KD-tree and collect new photons for each photon-gather request.
     * - Progressive photon mapping (https://dl.acm.org/doi/abs/10.1145/1457515.1409083)
     * - Update the squared gather radius (gather_radius_sq) and power, i.e. integrated irradiance, (gather_total_power) based on the new_photon_count and alpha parameter.
     * Hint: The KD-tree is addressed like a classical heap data structure:
     *       Given a node with index i, the left child is at index 2*i+1, and the right child at index 2*i+2.
     */

    // Stack of nodes used to traverse KD-tree
    opg::Stack<uint32_t, MAX_DEPTH> stack;
    unsigned int node = 0u; // Start at the root node

    uint32_t new_photon_count = 0u;
    glm::vec3 new_power = glm::vec3(0.0f);

    // TODO implement

    // SOLUTION

    stack.push(node);

    while(!stack.empty()) {
        uint32_t node_idx = stack.pop();

        const PhotonData& new_node = photon_map[node_idx];

        if (new_node.node_type == KDNodeType::Leaf){
            float dist_sq = glm::distance2(new_node.position, gather_position);
            if (dist_sq <= gather_radius_sq) {
                new_power += new_node.irradiance_weight * gather_throughput;
                new_photon_count++;
            }
        }
        else if (new_node.node_type == KDNodeType::DoubleLeaf) {
            // Two photons: check both
            for (int i = 0; i < 2; i++) {
                const PhotonData& photon = photon_map[node_idx + i];
                float dist_sq = glm::distance2(photon.position, gather_position);
                if (dist_sq <= gather_radius_sq) {
                    new_power += photon.irradiance_weight * gather_throughput;
                    new_photon_count++;
                }
            }
            // Skip the next node
            node_idx++;
        }
        else {
            
            // Determine split axis
            int split_axis;
            if (new_node.node_type == KDNodeType::AxisX) split_axis = 0;
            else if (new_node.node_type == KDNodeType::AxisY) split_axis = 1;
            else split_axis = 2; // AxisZ

            // Check if gather_position is on the left or right of the split
            bool go_left = (gather_position[split_axis] < new_node.position[split_axis]);

            // Push children onto stack: far child first, then near child
            uint32_t left_child = 2 * node_idx + 1;
            uint32_t right_child = 2 * node_idx + 2;
            stack.push(go_left ? right_child : left_child);
            stack.push(go_left ? left_child : right_child);

            // Also check the photon stored in this internal node
            float dist_sq = glm::distance2(new_node.position, gather_position);
            if (dist_sq <= gather_radius_sq) {
                new_power += new_node.irradiance_weight * gather_throughput;
                new_photon_count++;
            }
        }
    }

    // EO SOLUTION

    //

    // The photon map has a limited size, and some photons that *should* have been
    // stored in the photon map have been discarded.
    // In lieu of russian roulette we scale the contribution of the photons that made
    // it into the photon map.
    // E.g. if half the photons are stored, we double their contribution.
    new_power *= photon_map_store_count_ptr->desired_count / photon_map_store_count_ptr->actual_count;

    // Add newly accumulated power to power accumulated in previous subpasses
    gather_total_power += new_power;

    //
    // Apply progressive photon mapping
    //

    // TODO implement

    // SOLUTION

    uint32_t old_photon_count = gather_photon_count;
    gather_radius_sq = gather_radius_sq * (old_photon_count + alpha * new_photon_count) / (old_photon_count + new_photon_count);
    gather_photon_count = old_photon_count + new_photon_count;

    gather_total_power = (gather_total_power * gather_photon_count + new_power) / gather_photon_count;

    // // Compute output radiance (irradiance -> radiance)
    // output_radiance[gather_index] = new_total_power / (M_PIf); // * new_gather_radius_sq);

    // EO SOLUTION
    //

    // Compute gathered radiance from gathered irradiance?
    glm::vec3 gathered_radiance = gather_total_power / (glm::pi<float>() * gather_radius_sq * (*total_emitted_photon_count_ptr) );
    output_radiance[gather_index] = gathered_radiance;

    // Updat gather buffer
    gather_data[gather_index].gather_radius_sq = gather_radius_sq;
    gather_data[gather_index].photon_count = gather_photon_count;
    gather_data[gather_index].total_power = gather_total_power;
}

void gatherPhotons(
    opg::BufferView<PhotonData> photon_map,
    opg::BufferView<PhotonGatherData> gather_data,
    opg::BufferView<glm::vec3> output_radiance,
    PhotonMapStoreCount *photon_map_store_count_ptr,
    uint32_t *total_emitted_photon_count_ptr,
    float alpha
    )
{
    const int blockSize  = 512; // 512 is a size that works well with modern GPUs.
    const int blockCount = ceil_div<int>(gather_data.count, blockSize); // Spawn enough blocks such that each pair of elements is added in in its own thread.

    // Launch the kernel on the GPU!
    gatherPhotonsKernel<<<blockCount, blockSize>>>(photon_map, gather_data, output_radiance, photon_map_store_count_ptr, total_emitted_photon_count_ptr, alpha);
}


__global__ void combineOutputRadianceKernel(glm::uvec2 thread_count, opg::TensorView<glm::vec3, 2> output_tensor_view, opg::TensorView<glm::vec3, 2> accum_path_tensor_view, opg::TensorView<glm::vec3, 2> accum_photon_tensor_view)
{
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    if (thread_index.x >= thread_count.x || thread_index.y >= thread_count.y || thread_index.z >= 1)
        return;

    glm::uvec2 pixel_index = glm::xy(thread_index);

    glm::vec3 path_radiance = accum_path_tensor_view(pixel_index).value();
    glm::vec3 photon_radiance = accum_photon_tensor_view(pixel_index).value();
    glm::vec3 output_radiance = path_radiance + photon_radiance;

    output_tensor_view(pixel_index).value() = output_radiance;
}

void combineOutputRadiance(opg::TensorView<glm::vec3, 2> output_tensor_view, opg::TensorView<glm::vec3, 2> accum_path_tensor_view, opg::TensorView<glm::vec3, 2> accum_photon_tensor_view)
{
    uint32_t output_width = output_tensor_view.counts[1];
    uint32_t output_height = output_tensor_view.counts[0];
    glm::uvec2 thread_count(output_width, output_height);
    // Launch the kernel on the GPU!
    opg::launch_2d_kernel(combineOutputRadianceKernel, thread_count, output_tensor_view, accum_path_tensor_view, accum_photon_tensor_view);
}

