#pragma once

#include "opg/scene/interface/medium.h"
#include "opg/scene/components/grid.h"

#include "heterogeneousmedium.cuh"

class HeterogeneousMedium : public opg::Medium
{
public:
    HeterogeneousMedium(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~HeterogeneousMedium();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    opg::GridComponent*  m_grid;
    HeterogeneousMediumData m_medium_data;
    opg::DeviceBuffer<HeterogeneousMediumData> m_medium_data_buffer;
};
