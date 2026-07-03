#pragma once

#include "opg/scene/interface/phasefunction.h"

#include "phasefunctions.cuh"

class HenyeyGreensteinPhaseFunction : public opg::PhaseFunction
{
public:
    HenyeyGreensteinPhaseFunction(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~HenyeyGreensteinPhaseFunction();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    HenyeyGreensteinPhaseFunctionData m_phasefun_data;
    opg::DeviceBuffer<HenyeyGreensteinPhaseFunctionData> m_phasefun_data_buffer;
};
