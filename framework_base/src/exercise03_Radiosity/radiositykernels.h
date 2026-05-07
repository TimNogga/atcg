#pragma once

#include "opg/glmwrapper.h"

void launchJacobiIteration(
    int num_primitives,
    float lambda,
    const float* form_factor_matrix,
    const glm::vec3* emissions,
    const glm::vec3* albedos,
    const glm::vec3* b_current,
    glm::vec3* b_next
);