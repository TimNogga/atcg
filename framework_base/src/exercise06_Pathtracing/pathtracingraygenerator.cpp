#include "pathtracingraygenerator.h"

#include "opg/scene/scene.h"
#include "opg/opg.h"
#include "opg/scene/components/camera.h"
#include "opg/kernels/kernels.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

PathtracingRayGenerator::PathtracingRayGenerator(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    RayGenerator(std::move(_scene), _props),
    m_accum_buffer_height(0),
    m_accum_buffer_width(0),
    m_accum_sample_count(0),
    m_camera_revision(~0),
    m_subframe_index(0)
{
    m_launch_params_buffer.alloc(1);
}

PathtracingRayGenerator::~PathtracingRayGenerator()
{
}

void PathtracingRayGenerator::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    std::string ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "pathtracingraygenerator.cu");
    OptixProgramGroup raygen_prog_group         = pipeline->addRaygenShader({ptx_filename, "__raygen__main"});
    OptixProgramGroup surface_miss_prog_group   = pipeline->addMissShader({ptx_filename, "__miss__main"});
    OptixProgramGroup occlusion_miss_prog_group = pipeline->addMissShader({ptx_filename, "__miss__occlusion"});

    m_raygen_index         = sbt->addRaygenEntry(raygen_prog_group, nullptr);
    m_surface_miss_index   = sbt->addMissEntry(surface_miss_prog_group, nullptr);
    m_occlusion_miss_index = sbt->addMissEntry(occlusion_miss_prog_group, nullptr);
}

void PathtracingRayGenerator::finalize()
{
    std::vector<const EmitterVPtrTable *> emitter_vptr_tables;
    m_scene->traverseSceneComponents<opg::Emitter>([&](opg::Emitter *emitter){
        emitter_vptr_tables.push_back(emitter->getEmitterVPtrTable());
    });

    m_emitters_buffer.alloc(emitter_vptr_tables.size());
    m_emitters_buffer.upload(emitter_vptr_tables.data());
}

void PathtracingRayGenerator::launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer)
{
    // NOTE: We access tensors like numpy arrays.
    // 1st tensor dimension -> row -> y axis
    // 2nd tensor dimension -> column -> x axis
    uint32_t image_width = output_buffer.counts[1];
    uint32_t image_height = output_buffer.counts[0];

    // When the framebuffer resolution changes, or when the camera moves, we have to discard the previously accumulated samples and start anew
    if (m_accum_buffer_height != image_height || m_accum_buffer_width != image_width)
    {
        m_accum_buffer_width = image_width;
        m_accum_buffer_height = image_height;

        // Reallocate accum buffer on resize
        m_accum_buffer.alloc(m_accum_buffer_height * m_accum_buffer_width);

        // Reset sample count
        m_accum_sample_count = 0;
    }
    else if (m_camera->getRevision() != m_camera_revision)
    {
        // Reset sample count
        m_accum_sample_count = 0;
    }
    m_camera_revision = m_camera->getRevision();

    // Ensure that we have enough space to store the samples in the sample buffer
    m_sample_buffer.allocIfRequired(image_height * image_width);

    opg::TensorView<glm::vec3, 2> accum_tensor_view = opg::make_tensor_view<glm::vec3>(reinterpret_cast<glm::vec3*>(m_accum_buffer.data()), sizeof(glm::vec4), image_height, image_width);
    opg::TensorView<glm::vec3, 2> sample_tensor_view = opg::make_tensor_view<glm::vec3>(reinterpret_cast<glm::vec3*>(m_sample_buffer.data()), sizeof(glm::vec4), image_height, image_width);

    PathtracingLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;
    launch_params.subframe_index = m_subframe_index;

    launch_params.output_radiance = sample_tensor_view;
    launch_params.image_width = image_width;
    launch_params.image_height = image_height;

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = m_surface_miss_index;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = m_occlusion_miss_index;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    m_camera->getCameraData(launch_params.camera);

    launch_params.emitters = m_emitters_buffer.view();

    m_launch_params_buffer.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params_buffer.getRaw(), m_launch_params_buffer.byteSize(), sbt->getSBT(m_raygen_index), image_width, image_height, 1) );
    CUDA_SYNC_CHECK();

    // Accumulate new estimate and copy to output buffer.
    opg::accumulate_samples(sample_tensor_view.unsqueeze<0>(), accum_tensor_view, m_accum_sample_count);
    opg::copy(accum_tensor_view, output_buffer);

    // Advance subframe index and accumulated sample count.
    m_subframe_index++;
    m_accum_sample_count++;

    CUDA_SYNC_CHECK();
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(PathtracingRayGenerator, "raygen.pathtracing");

} // end namespace opg
