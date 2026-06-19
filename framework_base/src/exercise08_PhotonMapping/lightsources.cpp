#include "lightsources.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"

#include "common.h"
#include "opg/hostdevice/color.h"
#include "meshlightkernels.h"

PointLight::PointLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props)
{
    float intensity = _props.getFloat("intensity", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.intensity = intensity * color;
    m_data.position = glm::vec3(_props.getVector("position", glm::vec4(0)));
    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);
}

PointLight::~PointLight()
{
}

float PointLight::getTotalEmittedPower() const
{
    // Integrate intensity over sphere (of all directions) with surface area 4*pi
    return 4*glm::pi<float>() * rgb_to_scalar_weight(m_data.intensity);
}

void PointLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__pointlight_sampleLight" });
    OptixProgramGroup sample_photon_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__pointlight_samplePhoton" });

    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());
    uint32_t sample_photon_index = sbt->addCallableEntry(sample_photon_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = +EmitterFlag::InfinitesimalSize;
    vptr_table_data.emitter_weight = getTotalEmittedPower();
    vptr_table_data.sampleCallIndex = sample_light_index;
    vptr_table_data.samplePhotonCallIndex = sample_photon_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}


DirectionalLight::DirectionalLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props)
{
    float irradiance = _props.getFloat("irradiance", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.irradiance_at_receiver = irradiance * color;
    m_data.direction = glm::normalize(glm::vec3(_props.getVector("direction", glm::vec4(0, 0, 1, 0))));
    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);
}

DirectionalLight::~DirectionalLight()
{
}

void DirectionalLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__directionallight_sampleLight" });
    OptixProgramGroup sample_photon_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__directionallight_samplePhoton" });

    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());
    uint32_t sample_photon_index = sbt->addCallableEntry(sample_photon_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = EmitterFlag::InfinitesimalSize | EmitterFlag::DistantEmitter;
    vptr_table_data.emitter_weight = getTotalEmittedPower();
    vptr_table_data.sampleCallIndex = sample_light_index;
    vptr_table_data.samplePhotonCallIndex = sample_photon_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}

SphereLight::SphereLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props),
    m_shape(0)
{
    float radiance = _props.getFloat("radiance", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.radiance = radiance * color;
    m_data_buffer.alloc(1);
    // Buffer upload happens later _after_ the shape instance is known
}

SphereLight::~SphereLight()
{
}

void SphereLight::assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance)
{
    if (m_shape != nullptr)
    {
        throw std::runtime_error("SphereLight can only be used in a single ShapeInstance!");
    }
    m_shape = dynamic_cast<opg::SphereShape*>(instance->getShape());
    if (m_shape == nullptr)
    {
        throw std::runtime_error("SphereLight needs to be assigned to a SphereShape!");
    }

    glm::mat4 transform = instance->getTransform();
    // transform has to be translation + uniform scaling, nothing more!

    m_data.position = glm::vec3(transform[3]);
    m_data.radius = transform[0][0];

    m_data_buffer.upload(&m_data);
}

float SphereLight::getTotalEmittedPower() const
{
    // Integrate radiance over hemisphere (of directions) weighted with cos(theta)
    float direction_integral = 0.5f;
    // Integrate radiance over sphere (of positions) with area 4*pi*r**2
    float area_integral = 4*glm::pi<float>()*m_data.radius*m_data.radius;

    return direction_integral * area_integral * rgb_to_scalar_weight(m_data.radiance);
}

void SphereLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup eval_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__spherelight_evalLight" });
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__spherelight_sampleLight" });
    OptixProgramGroup eval_light_sampling_pdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__spherelight_evalLightSamplingPDF" });
    OptixProgramGroup sample_photon_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__spherelight_samplePhoton" });

    uint32_t eval_light_index = sbt->addCallableEntry(eval_light_prog_group, m_data_buffer.data());
    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());
    uint32_t eval_light_sampling_pdf_index = sbt->addCallableEntry(eval_light_sampling_pdf_prog_group, m_data_buffer.data());
    uint32_t sample_photon_index = sbt->addCallableEntry(sample_photon_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = 0;
    vptr_table_data.emitter_weight = getTotalEmittedPower();
    vptr_table_data.evalCallIndex = eval_light_index;
    vptr_table_data.sampleCallIndex = sample_light_index;
    vptr_table_data.evalSamplingPdfCallIndex = eval_light_sampling_pdf_index;
    vptr_table_data.samplePhotonCallIndex = sample_photon_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}



MeshLight::MeshLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props),
    m_shape(0)
{
    float radiance = _props.getFloat("radiance", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.radiance = radiance * color;
    m_data.double_sided = _props.getBool("double_sided", true);
    m_data_buffer.alloc(1);
    // Buffer upload happens later _after_ the shape instance is known
}

MeshLight::~MeshLight()
{
}

void MeshLight::assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance)
{
    if (m_shape != nullptr)
    {
        throw std::runtime_error("MeshLight can only be used in a single ShapeInstance!");
    }
    m_shape = dynamic_cast<opg::MeshShape*>(instance->getShape());
    if (m_shape == nullptr)
    {
        throw std::runtime_error("MeshLight needs to be assigned to a MeshShape!");
    }

    m_data.local_to_world = instance->getTransform();
    m_data.world_to_local = glm::inverse(m_data.local_to_world);

    m_data.mesh_indices = m_shape->getMeshShapeData().indices;
    m_data.mesh_positions = m_shape->getMeshShapeData().positions;

    m_mesh_cdf_buffer.alloc(m_data.mesh_indices.count);
    m_data.mesh_cdf = m_mesh_cdf_buffer.view();

    computeMeshTrianglePDF(m_data.local_to_world, m_data.mesh_indices, m_data.mesh_positions, m_data.mesh_cdf);
    CUDA_SYNC_CHECK();
    computeMeshTriangleCDF(m_data.mesh_cdf);
    CUDA_SYNC_CHECK();
    CUDA_CHECK( cudaMemcpy(&m_data.total_surface_area, &m_data.mesh_cdf[m_data.mesh_cdf.count-1], sizeof(float), cudaMemcpyDeviceToHost) );
    normalizeMeshTriangleCDF(m_data.mesh_cdf, m_data.total_surface_area);

    // If the mesh is double-sided the total surface area is twice as large!
    if (m_data.double_sided)
        m_data.total_surface_area *= 2;

    m_data_buffer.upload(&m_data);
}

float MeshLight::getTotalEmittedPower() const
{
    // Integrate radiance over (hemi)sphere (of directions) weighted with cos(theta)
    float direction_integral = 1.0f; // 0.5f; 1.0 for double-sided, 0.5 for single-sided
    // Integrate radiance over mesh surface
    float area_integral = m_data.total_surface_area;

    return direction_integral * area_integral * rgb_to_scalar_weight(m_data.radiance);
}

void MeshLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup eval_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__meshlight_evalLight" });
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__meshlight_sampleLight" });
    OptixProgramGroup eval_light_sampling_pdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__meshlight_evalLightSamplingPDF" });
    OptixProgramGroup sample_photon_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__meshlight_samplePhoton" });

    uint32_t eval_light_index = sbt->addCallableEntry(eval_light_prog_group, m_data_buffer.data());
    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());
    uint32_t eval_light_sampling_pdf_index = sbt->addCallableEntry(eval_light_sampling_pdf_prog_group, m_data_buffer.data());
    uint32_t sample_photon_index = sbt->addCallableEntry(sample_photon_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = 0;
    vptr_table_data.emitter_weight = getTotalEmittedPower();
    vptr_table_data.evalCallIndex = eval_light_index;
    vptr_table_data.sampleCallIndex = sample_light_index;
    vptr_table_data.evalSamplingPdfCallIndex = eval_light_sampling_pdf_index;
    vptr_table_data.samplePhotonCallIndex = sample_photon_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}



namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(PointLight, "emitter.point");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(DirectionalLight, "emitter.directional");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(SphereLight, "emitter.sphere");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(MeshLight, "emitter.mesh");

} // end namespace opg
