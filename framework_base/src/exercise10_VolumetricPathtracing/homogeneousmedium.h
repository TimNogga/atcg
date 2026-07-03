#pragma once

#include "opg/scene/interface/medium.h"

#include "homogeneousmedium.cuh"

class HomogeneousMedium : public opg::Medium
{
public:
    HomogeneousMedium(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~HomogeneousMedium();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    HomogeneousMediumData m_medium_data;
    opg::DeviceBuffer<HomogeneousMediumData> m_medium_data_buffer;
};
