#include "mesh.h"
#include "opg/scene/scene.h"
#include "opg/scene/interface/bsdf.h"
#include "opg/scene/interface/emitter.h"
#include "opg/scene/components/bufferviewcomponent.h"

#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/exception.h"
#include "opg/opg.h"

#include "opg/raytracing/opg_optix_stubs.h"


MeshShape::MeshShape(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Shape(std::move(_scene), _props)
{
    if (opg::BufferViewComponent *view_comp = _props.getComponentAs<opg::BufferViewComponent>("indices"))
    {
        m_data.indices   = view_comp->getBufferView();
    }
    if (opg::BufferViewComponent *view_comp = _props.getComponentAs<opg::BufferViewComponent>("positions"))
    {
        m_data.positions = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (opg::BufferViewComponent *view_comp = _props.getComponentAs<opg::BufferViewComponent>("normals"))
    {
        m_data.normals   = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (opg::BufferViewComponent *view_comp = _props.getComponentAs<opg::BufferViewComponent>("tangents"))
    {
        m_data.tangents  = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (opg::BufferViewComponent *view_comp = _props.getComponentAs<opg::BufferViewComponent>("uvs"))
    {
        m_data.uvs       = view_comp->getBufferView().asType<glm::vec2>();
    }

    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);

    m_data_ptr = m_data_buffer.data();
}

MeshShape::~MeshShape()
{
}

void MeshShape::prepareAccelerationStructure()
{
    uint32_t geometry_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE; // OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_input = {};
    build_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes         =
        m_data.positions.byte_stride ?
        m_data.positions.byte_stride :
        sizeof(decltype(m_data.positions[0]));
    build_input.triangleArray.numVertices                 = m_data.positions.count;
    build_input.triangleArray.vertexBuffers               = reinterpret_cast<CUdeviceptr*>(&m_data.positions.data);
    build_input.triangleArray.indexFormat                 =
        m_data.indices.elmt_byte_size == sizeof(glm::u16vec3) ?
        OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :  // glm::u16vec3
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;     // glm::u32vec3
    build_input.triangleArray.indexStrideInBytes          = m_data.indices.byte_stride;
    build_input.triangleArray.numIndexTriplets            = m_data.indices.count;
    build_input.triangleArray.indexBuffer                 = reinterpret_cast<CUdeviceptr>(m_data.indices.data);
    build_input.triangleArray.flags                       = &geometry_input_flags;
    build_input.triangleArray.numSbtRecords               = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(m_scene->getContext(), &accel_options, &build_input, 1, &gas_buffer_sizes) );

    opg::GenericDeviceBuffer d_temp(gas_buffer_sizes.tempSizeInBytes);

    m_ast_buffer.alloc(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK( optixAccelBuild(
        m_scene->getContext(), 0,   // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp.getRaw(),
        d_temp.byteSize(),
        m_ast_buffer.getRaw(),
        m_ast_buffer.byteSize(),
        &m_ast_handle,
        nullptr,
        0
    ) );
}

void MeshShape::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "mesh.cu");
    m_hit_prog_group = pipeline->addTrianglesHitGroupShader({ ptx_filename, "__closesthit__mesh" }, { ptx_filename, "__anyhit__mesh" });
}
