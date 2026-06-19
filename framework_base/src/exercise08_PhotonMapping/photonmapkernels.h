#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "photonmapdata.cuh"

void resetPhotonGatherData(opg::BufferView<PhotonGatherData> gather_data, float gather_radius_sq);

void gatherPhotons(
    opg::BufferView<PhotonData> photon_map,
    opg::BufferView<PhotonGatherData> gather_data,
    opg::BufferView<glm::vec3> output_radiance,
    PhotonMapStoreCount *photon_map_store_count_ptr,
    uint32_t *total_emitted_photon_count_ptr,
    float alpha
    );

// Combine the result due to path tracing and photon mapping into the output buffer.
void combineOutputRadiance(opg::TensorView<glm::vec3, 2> output_tensor_view, opg::TensorView<glm::vec3, 2> accum_path_tensor_view, opg::TensorView<glm::vec3, 2> accum_photon_tensor_view);
