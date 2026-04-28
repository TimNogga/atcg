#pragma once

#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "cylinder.cuh"

class CylinderShape : public opg::Shape
{
public:
    CylinderShape(PrivatePtr<opg::Scene> scene, const opg::Properties &_props);
    virtual ~CylinderShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;
};
