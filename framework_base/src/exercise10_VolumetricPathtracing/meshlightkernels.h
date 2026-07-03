#pragma once

#include "opg/memory/bufferview.h"
#include "opg/glmwrapper.h"

void computeMeshTrianglePDF(const glm::mat4 &local_to_world, const opg::GenericBufferView &mesh_indices, const opg::BufferView<glm::vec3> &mesh_positions, opg::BufferView<float> &pdf);
void computeMeshTriangleCDF(opg::BufferView<float> &cdf);
void normalizeMeshTriangleCDF(opg::BufferView<float> &cdf, float total_value);
