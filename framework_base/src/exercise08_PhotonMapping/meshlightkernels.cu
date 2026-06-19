#include <cuda_runtime.h>
#include <optix.h>

#include "meshlightkernels.h"
#include "opg/hostdevice/misc.h"

#pragma cuda_source_property_format=OBJ

__global__ void computeMeshTrianglePDF_kernel(const glm::mat4 local_to_world, const opg::GenericBufferView mesh_indices, const opg::BufferView<glm::vec3> mesh_positions, opg::BufferView<float> pdf)
{
    uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= pdf.count)
        return;


    // Indices of triangle vertices in the mesh
    glm::uvec3 vertex_indices = glm::uvec3(0u);
    if (mesh_indices.elmt_byte_size == sizeof(glm::u32vec3))
    {
        // Indices stored as 32-bit unsigned integers
        const glm::u32vec3* indices = reinterpret_cast<glm::u32vec3*>(mesh_indices.data);
        vertex_indices = glm::uvec3(indices[thread_index]);
    }
    else
    {
        // Indices stored as 16-bit unsigned integers
        const glm::u16vec3* indices = reinterpret_cast<glm::u16vec3*>(mesh_indices.data);
        vertex_indices = glm::uvec3(indices[thread_index]);
    }

    // Vertex positions of selected triangle
    glm::vec3 P0 = glm::vec3(local_to_world * glm::vec4(mesh_positions[vertex_indices.x], 1));
    glm::vec3 P1 = glm::vec3(local_to_world * glm::vec4(mesh_positions[vertex_indices.y], 1));
    glm::vec3 P2 = glm::vec3(local_to_world * glm::vec4(mesh_positions[vertex_indices.z], 1));

    // Compute triangle area
    float parallelogram_area = glm::length(glm::cross(P1-P0, P2-P0));
    float triangle_area = 0.5f * parallelogram_area;

    // Write unnormalized pdf
    pdf[thread_index] = triangle_area;
}

void computeMeshTrianglePDF(const glm::mat4 &local_to_world, const opg::GenericBufferView &mesh_indices, const opg::BufferView<glm::vec3> &mesh_positions, opg::BufferView<float> &pdf)
{
    uint32_t blockSize = 512;
    uint32_t blockCount = ceil_div<uint32_t>(pdf.count, blockSize);
    computeMeshTrianglePDF_kernel<<<blockCount, blockSize>>>(local_to_world, mesh_indices, mesh_positions, pdf);
}

__global__ void computeMeshTriangleCDF_kernel(opg::BufferView<float> cdf)
{
    float acc = 0;
    for (uint32_t i = 0; i < cdf.count; ++i)
    {
        acc += cdf[i];
        cdf[i] = acc;
    }
}

void computeMeshTriangleCDF(opg::BufferView<float> &cdf)
{
    uint32_t blockSize = 1;
    uint32_t blockCount = 1;
    computeMeshTriangleCDF_kernel<<<blockCount, blockSize>>>(cdf);
}

__global__ void normalizeMeshTriangleCDF_kernel(opg::BufferView<float> cdf, float total_value)
{
    uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= cdf.count)
        return;

    cdf[thread_index] /= total_value;
}

void normalizeMeshTriangleCDF(opg::BufferView<float> &cdf, float total_value)
{
    uint32_t blockSize = 512;
    uint32_t blockCount = ceil_div<uint32_t>(cdf.count, blockSize);
    normalizeMeshTriangleCDF_kernel<<<blockCount, blockSize>>>(cdf, total_value);
}
