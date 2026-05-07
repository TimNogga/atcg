#include "radiositykernels.h"
#include "opg/hostdevice/misc.h"

#pragma cuda_source_property_format=OBJ

__global__ void jacobiKernel(
    int N, 
    float lambda,
    const float* F,
    const glm::vec3* E,
    const glm::vec3* r,
    const glm::vec3* b_curr,
    glm::vec3* b_next)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    glm::vec3 gathered_light = glm::vec3(0.0f);
    for (int j = 0; j < N; ++j) 
    {
        float F_ij = F[i * N + j];
        gathered_light += F_ij * b_curr[j];
    }

    b_next[i] = b_curr[i] + lambda * (E[i] - b_curr[i] + r[i] * gathered_light);
}

void launchJacobiIteration(
    int num_primitives,
    float lambda,
    const float* form_factor_matrix,
    const glm::vec3* emissions,
    const glm::vec3* albedos,
    const glm::vec3* b_current,
    glm::vec3* b_next)
{
    int blockSize = 256;
    int numBlocks = (num_primitives + blockSize - 1) / blockSize;
    
    jacobiKernel<<<numBlocks, blockSize>>>(
        num_primitives, lambda, form_factor_matrix, emissions, albedos, b_current, b_next
    );
}