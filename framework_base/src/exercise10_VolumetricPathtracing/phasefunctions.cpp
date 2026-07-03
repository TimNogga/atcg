#include "phasefunctions.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"


HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    PhaseFunction(std::move(_scene), _props)
{
    m_phasefun_data.g = _props.getFloat("g", 0.0f);

    m_phasefun_data_buffer.alloc(1);
    m_phasefun_data_buffer.upload(&m_phasefun_data);
}

HenyeyGreensteinPhaseFunction::~HenyeyGreensteinPhaseFunction()
{
}

void HenyeyGreensteinPhaseFunction::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "phasefunctions.cu");
    OptixProgramGroup eval_phase_function_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__henyeygreenstein_evalPhaseFunction" });
    OptixProgramGroup sample_phase_function_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__henyeygreenstein_samplePhaseFunction" });

    uint32_t eval_phase_function_index = sbt->addCallableEntry(eval_phase_function_prog_group, m_phasefun_data_buffer.data());
    uint32_t sample_phase_function_index = sbt->addCallableEntry(sample_phase_function_prog_group, m_phasefun_data_buffer.data());

    PhaseFunctionVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex = eval_phase_function_index;
    vptr_table_data.sampleCallIndex = sample_phase_function_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(HenyeyGreensteinPhaseFunction, "phase_function.henyey_greenstein");

} // end namespace opg
