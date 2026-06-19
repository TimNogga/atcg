#pragma once

#include "opg/scene/interface/bsdf.h"

#include "bsdfmodels.cuh"

class DiffuseBSDF : public opg::BSDF
{
public:
    DiffuseBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~DiffuseBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    DiffuseBSDFData m_bsdf_data;
    opg::DeviceBuffer<DiffuseBSDFData> m_bsdf_data_buffer;
};

class SpecularBSDF : public opg::BSDF
{
public:
    SpecularBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~SpecularBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    SpecularBSDFData m_bsdf_data;
    opg::DeviceBuffer<SpecularBSDFData> m_bsdf_data_buffer;
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
