#include "homogeneousmedium.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"


HomogeneousMedium::HomogeneousMedium(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Medium(std::move(_scene), _props)
{
    //m_medium_data.sigma_t = _props.getFloat("sigma_t", 1.0f);
    //m_medium_data.albedo = _props.getVector("albedo", glm::vec4(1));
    m_medium_data.sigma_s = _props.getVector("sigma_s", glm::vec4(0));
    m_medium_data.sigma_a = _props.getVector("sigma_a", glm::vec4(0));

    m_medium_data_buffer.alloc(1);
    m_medium_data_buffer.upload(&m_medium_data);
}

HomogeneousMedium::~HomogeneousMedium()
{
}

void HomogeneousMedium::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
   if (m_phase_function != nullptr)
        m_phase_function->ensurePipelineInitialized(pipeline, sbt);

    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "homogeneousmedium.cu");
    OptixProgramGroup eval_transmittance_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance" });
    OptixProgramGroup sample_medium_event_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent" });

    uint32_t eval_transmittance_index = sbt->addCallableEntry(eval_transmittance_prog_group, m_medium_data_buffer.data());
    uint32_t sample_medium_event_index = sbt->addCallableEntry(sample_medium_event_prog_group, m_medium_data_buffer.data());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function = m_phase_function ? m_phase_function->getPhaseFunctionVPtrTable() : nullptr;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}


namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(HomogeneousMedium, "medium.homogeneous");

} // end namespace opg
