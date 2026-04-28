#pragma once

#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "mesh.cuh"

class MeshShape : public opg::Shape
{
public:
    MeshShape(PrivatePtr<opg::Scene> scene, const opg::Properties &_props);
    virtual ~MeshShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    MeshShapeData                       m_data;
    opg::DeviceBuffer<MeshShapeData>    m_data_buffer;
};
