#pragma once

#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "sphere.cuh"

class SphereShape : public opg::Shape
{
public:
    SphereShape(PrivatePtr<opg::Scene> scene, const opg::Properties &_props);
    virtual ~SphereShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;
};
