#pragma once

#include "opg/scene/interface/raygenerator.h"

#include "raycastingraygenerator.cuh"

class RayCastingRayGenerator : public opg::RayGenerator
{
public:
    RayCastingRayGenerator(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    ~RayCastingRayGenerator();

    virtual void launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer) override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    opg::DeviceBuffer<RayCastingLaunchParams> m_launch_params;
};
