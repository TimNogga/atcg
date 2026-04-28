#include "raycastingraygenerator.h"

#include "opg/scene/scene.h"
#include "opg/opg.h"
#include "opg/scene/components/camera.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

RayCastingRayGenerator::RayCastingRayGenerator(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    RayGenerator(std::move(_scene), _props)
{
    m_launch_params.alloc(1);
}

RayCastingRayGenerator::~RayCastingRayGenerator()
{
}

void RayCastingRayGenerator::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    OptixProgramGroup progGroup = pipeline->addRaygenShader({opg::getPtxFilename(OPG_TARGET_NAME, "raycastingraygenerator.cu"), "__raygen__main"});
    OptixProgramGroup missProgGroup = pipeline->addMissShader({opg::getPtxFilename(OPG_TARGET_NAME, "raycastingraygenerator.cu"), "__miss__main"});

    sbt->addRaygenEntry(progGroup, nullptr);
    sbt->addMissEntry(missProgGroup, nullptr);
}

void RayCastingRayGenerator::launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer)
{
    // NOTE: We access tensors like numpy arrays.
    // 1st tensor dimension -> row -> y axis
    // 2nd tensor dimension -> column -> x axis
    uint32_t image_width = output_buffer.counts[1];
    uint32_t image_height = output_buffer.counts[0];

    RayCastingLaunchParams launch_params;
    launch_params.output_buffer = output_buffer;
    launch_params.image_width = image_width;
    launch_params.image_height = image_height;
    launch_params.traversable_handle = m_scene->getTraversableHandle(1);
    m_camera->getCameraData(launch_params.camera);

    launch_params.traceParams.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.traceParams.SBToffset = 0;
    launch_params.traceParams.SBTstride = 1;
    launch_params.traceParams.missSBTIndex = 0;

    m_launch_params.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params.getRaw(), m_launch_params.byteSize(), sbt->getSBT(), image_width, image_height, 1) );
    CUDA_SYNC_CHECK();
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(RayCastingRayGenerator, "raygen.raycasting");

} // end namespace opg
