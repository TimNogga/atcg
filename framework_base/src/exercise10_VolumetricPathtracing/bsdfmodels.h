#pragma once

#include "opg/scene/interface/bsdf.h"

#include "bsdfmodels.cuh"

class GGXBSDF : public opg::BSDF
{
public:
    GGXBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~GGXBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    GGXBSDFData m_bsdf_data;
    opg::DeviceBuffer<GGXBSDFData> m_bsdf_data_buffer;
};

class RefractiveBSDF : public opg::BSDF
{
public:
    RefractiveBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~RefractiveBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    RefractiveBSDFData m_bsdf_data;
    opg::DeviceBuffer<RefractiveBSDFData> m_bsdf_data_buffer;
};

class NullBSDF : public opg::BSDF
{
public:
    NullBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~NullBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;
};
