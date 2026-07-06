#pragma once

#include "opg/scene/interface/emitter.h"
#include "opg/scene/components/shapeinstance.h"
#include "opg/scene/components/shapes/sphere.h"
#include "opg/scene/components/shapes/mesh.h"
#include "opg/memory/devicebuffer.h"

#include "lightsources.cuh"

class PointLight : public opg::Emitter
{
public:
    PointLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~PointLight();

    virtual float getTotalEmittedPower() const override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    PointLightData m_data;
    opg::DeviceBuffer<PointLightData> m_data_buffer;
};

class DirectionalLight : public opg::Emitter
{
public:
    DirectionalLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~DirectionalLight();

    // virtual float getTotalEmittedPower() const override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    DirectionalLightData m_data;
    opg::DeviceBuffer<DirectionalLightData> m_data_buffer;
};


class SphereLight : public opg::Emitter
{
public:
    SphereLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~SphereLight();

    virtual void assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance) override;

    virtual float getTotalEmittedPower() const override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    opg::SphereShape *m_shape;

    SphereLightData m_data;
    opg::DeviceBuffer<SphereLightData> m_data_buffer;
};

class MeshLight : public opg::Emitter
{
public:
    MeshLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~MeshLight();

    virtual void assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance) override;

    virtual float getTotalEmittedPower() const override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    opg::MeshShape *m_shape;

    opg::DeviceBuffer<float> m_mesh_cdf_buffer;

    MeshLightData m_data;
    opg::DeviceBuffer<MeshLightData> m_data_buffer;
};
