#include "heterogeneousmedium.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"


HeterogeneousMedium::HeterogeneousMedium(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Medium(std::move(_scene), _props),
    m_grid { _props.getComponentAs<opg::GridComponent>("grid") }
{
    glm::mat4 to_world = _props.getMatrix("to_world", glm::mat4(1));
    m_medium_data.world_to_local = glm::inverse(to_world);

    m_medium_data.density_scale = _props.getFloat("density_scale", 1.0f);

    m_medium_data.density_majorant = m_grid->getMaximum() * m_medium_data.density_scale;

    m_medium_data.density_grid = m_grid->getGridData();

    m_medium_data_buffer.alloc(1);
    m_medium_data_buffer.upload(&m_medium_data);
}

HeterogeneousMedium::~HeterogeneousMedium()
{
}

void HeterogeneousMedium::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
   if (m_phase_function != nullptr)
        m_phase_function->ensurePipelineInitialized(pipeline, sbt);

    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "heterogeneousmedium.cu");
    OptixProgramGroup eval_transmittance_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__heterogeneousMedium_evalTransmittance" });
    OptixProgramGroup sample_medium_event_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__heterogeneousMedium_sampleMediumEvent" });

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

OPG_REGISTER_SCENE_COMPONENT_FACTORY(HeterogeneousMedium, "medium.heterogeneous");

} // end namespace opg
