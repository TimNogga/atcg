#pragma once

#include "opg/scene/interface/raygenerator.h"
#include "opg/scene/interface/emitter.h"

#include "photonmappingraygenerator.cuh"

class PhotonMappingRayGenerator : public opg::RayGenerator
{
public:
    PhotonMappingRayGenerator(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    ~PhotonMappingRayGenerator();

    virtual void launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer) override;

    virtual void finalize() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    void launchTraceImage(CUstream stream);
    void launchTracePhotons(CUstream stream);
    void launchGatherPhotons(CUstream stream);

    void buildPhotonMapKDTree(CUstream stream);

private:
    // This buffer is filled with the launch parameters before each ray tracing launch.
    opg::DeviceBuffer<PhotonMappingLaunchParams>  m_launch_params_buffer;

    // List of all emitters that are used for direct illumination on the GPU.
    float                                       m_emitters_total_weight;
    opg::DeviceBuffer<float>                    m_emitters_cdf_buffer;
    opg::DeviceBuffer<const EmitterVPtrTable*>  m_emitters_buffer;


    // Photon map related buffers
    opg::DeviceBuffer<PhotonData> m_photon_map_buffer;
    opg::DeviceBuffer<PhotonMapStoreCount> m_photon_store_count_buffer; // photon_store_count
    opg::DeviceBuffer<uint32_t> m_photon_emitted_count_buffer; // photon_emitted_count

    // Per-pixel buffers related to photons
    opg::DeviceBuffer<PhotonGatherData> m_photon_gather_buffer;
    opg::DeviceBuffer<glm::vec4> m_accum_buffer_photons;

    // Store the one-sample-per-pixel estimate for the current subframe.
    opg::DeviceBuffer<glm::vec4> m_sample_buffer;

    // In every frame one ray is randomly shot from the camera.
    // The result of the individual rays is averaged here over multiple frames.
    opg::DeviceBuffer<glm::vec4> m_accum_buffer_path;
    uint32_t m_accum_buffer_width;
    uint32_t m_accum_buffer_height;

    // Count how many frames we have rendered from the current camera view.
    // This is important to generate different rays in every ray tracing launch, and also for updating the accumulation buffer.
    uint32_t m_accum_sample_count;

    // The "revision" of the camera changes every time it is modified.
    // Checking if the revision changed is easier than to check for changes in individual parameters.
    uint32_t m_camera_revision;

    // Used to seed the random number generator differently in each subframe. The subframe index is never reset.
    uint32_t m_subframe_index       = 0;

    // The number of thread that are used to spawn new photons in the photon map.
    // Set this to a value that just saturates your GPU, don't choose to large.
    uint32_t m_photon_thread_count;

    // The initial radius used for photon gather operations.
    float m_gather_radius;
    // The alpha parameter of the progressive photon mapping algorithm that controls how fast the gather radius is shrunk (1 = do not shrink).
    float m_gather_alpha;



    // Indices of shader programs in the shader binding table
    uint32_t m_trace_lights_raygen_index   = 0;
    uint32_t m_trace_camera_raygen_index   = 0;
    uint32_t m_surface_miss_index          = 0;
    uint32_t m_occlusion_miss_index        = 0;
};
