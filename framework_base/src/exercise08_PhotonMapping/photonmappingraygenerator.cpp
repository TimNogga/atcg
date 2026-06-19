#include "photonmappingraygenerator.h"

#include "photonmapkernels.h"
#include "select.h"

#include "opg/scene/scene.h"
#include "opg/opg.h"
#include "opg/scene/components/camera.h"
#include "opg/kernels/kernels.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

#include <algorithm>
#include <numeric>

#include <iostream>

PhotonMappingRayGenerator::PhotonMappingRayGenerator(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    RayGenerator(std::move(_scene), _props),
    m_subframe_index(0),
    m_accum_buffer_height(0),
    m_accum_buffer_width(0),
    m_accum_sample_count(0),
    m_camera_revision(~0)
{
    uint32_t photon_map_size = _props.getInt("photon_map_size", 128*1024);
    m_photon_thread_count = _props.getInt("photon_thread_count", 8192);
    m_gather_alpha = _props.getFloat("gather_alpha", 0.7f);
    m_gather_radius = _props.getFloat("gather_radius", 0.5f);

    // Make sure that the photon map size is actually odd to avoid nodes with single childs in kd tree!
    photon_map_size |= 1;
    m_photon_map_buffer.alloc(photon_map_size);

    m_photon_store_count_buffer.alloc(1);
    m_photon_emitted_count_buffer.alloc(1);

    m_launch_params_buffer.alloc(1);
}

PhotonMappingRayGenerator::~PhotonMappingRayGenerator()
{
}

void PhotonMappingRayGenerator::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    std::string ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "photonmappingraygenerator.cu");
    OptixProgramGroup trace_lights_raygen_prog_group = pipeline->addRaygenShader({ptx_filename, "__raygen__traceLights"});
    OptixProgramGroup trace_camera_raygen_prog_group = pipeline->addRaygenShader({ptx_filename, "__raygen__traceCamera"});
    OptixProgramGroup surface_miss_prog_group        = pipeline->addMissShader({ptx_filename, "__miss__main"});
    OptixProgramGroup occlusion_miss_prog_group      = pipeline->addMissShader({ptx_filename, "__miss__occlusion"});

    m_trace_lights_raygen_index = sbt->addRaygenEntry(trace_lights_raygen_prog_group, nullptr);
    m_trace_camera_raygen_index = sbt->addRaygenEntry(trace_camera_raygen_prog_group, nullptr);
    m_surface_miss_index        = sbt->addMissEntry(surface_miss_prog_group, nullptr);
    m_occlusion_miss_index      = sbt->addMissEntry(occlusion_miss_prog_group, nullptr);
}

void PhotonMappingRayGenerator::finalize()
{
    std::vector<const EmitterVPtrTable *> emitter_vptr_tables;
    std::vector<float> emitters_cdf;
    m_scene->traverseSceneComponents<opg::Emitter>([&](opg::Emitter *emitter){
        emitter_vptr_tables.push_back(emitter->getEmitterVPtrTable());
        emitters_cdf.push_back(emitter->getTotalEmittedPower());
    });

    // Compute the cdf over emitters on the CPU.
    // This is easier and there usually are not that many light sources either...
    std::partial_sum(emitters_cdf.begin(), emitters_cdf.end(), emitters_cdf.begin());
    m_emitters_total_weight = emitters_cdf.back();
    for (float &cdf_value : emitters_cdf)
        cdf_value /= emitters_cdf.back();

    // Upload emitter cdfs
    m_emitters_cdf_buffer.alloc(emitters_cdf.size());
    m_emitters_cdf_buffer.upload(emitters_cdf.data());

    // Upload emitter vptr tables
    m_emitters_buffer.alloc(emitter_vptr_tables.size());
    m_emitters_buffer.upload(emitter_vptr_tables.data());
}


void PhotonMappingRayGenerator::launchTraceImage(CUstream stream)
{
    opg::TensorView<glm::vec3, 2> accum_tensor_view = opg::make_tensor_view<glm::vec3>(reinterpret_cast<glm::vec3*>(m_accum_buffer_path.data()), sizeof(glm::vec4), m_accum_buffer_height, m_accum_buffer_width);
    opg::TensorView<glm::vec3, 2> sample_tensor_view = opg::make_tensor_view<glm::vec3>(reinterpret_cast<glm::vec3*>(m_sample_buffer.data()), sizeof(glm::vec4), m_accum_buffer_height, m_accum_buffer_width);

    PhotonMappingLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;
    launch_params.subframe_index = m_subframe_index;

    launch_params.image_params.photon_gather_positions = opg::make_tensor_view<glm::vec3>(
            m_photon_gather_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonGatherData, PhotonGatherData::position)),
            m_accum_buffer_height, m_accum_buffer_width);

    launch_params.image_params.photon_gather_normals = opg::make_tensor_view<glm::vec3>(
            m_photon_gather_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonGatherData, PhotonGatherData::normal)),
            m_accum_buffer_height, m_accum_buffer_width);

    launch_params.image_params.photon_gather_throughputs = opg::make_tensor_view<glm::vec3>(
            m_photon_gather_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonGatherData, PhotonGatherData::throughput)),
            m_accum_buffer_height, m_accum_buffer_width);

    launch_params.image_params.output_radiance = sample_tensor_view;
    launch_params.image_params.image_width = m_accum_buffer_width;
    launch_params.image_params.image_height = m_accum_buffer_height;

    m_camera->getCameraData(launch_params.image_params.camera);

    launch_params.emitters_total_weight = m_emitters_total_weight;
    launch_params.emitters_cdf = m_emitters_cdf_buffer.view();
    launch_params.emitters = m_emitters_buffer.view();

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = m_surface_miss_index;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = m_occlusion_miss_index;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    m_launch_params_buffer.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params_buffer.getRaw(), m_launch_params_buffer.byteSize(), sbt->getSBT(m_trace_camera_raygen_index), m_accum_buffer_width, m_accum_buffer_height, 1) );
    CUDA_SYNC_CHECK();

    // Accumulate new estimate and copy to output buffer.
    opg::accumulate_samples(sample_tensor_view.unsqueeze<0>(), accum_tensor_view, m_accum_sample_count);
    CUDA_SYNC_CHECK();
}

void PhotonMappingRayGenerator::launchTracePhotons(CUstream stream)
{
    PhotonMappingLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;
    launch_params.subframe_index = m_subframe_index;

    launch_params.photon_params.photon_positions = m_photon_map_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonData, PhotonData::position));
    launch_params.photon_params.photon_normals = m_photon_map_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonData, PhotonData::normal));
    launch_params.photon_params.photon_irradiance_weights = m_photon_map_buffer.view().reinterpret<glm::vec3>(offsetof(PhotonData, PhotonData::irradiance_weight));

    launch_params.photon_params.photon_store_count = m_photon_store_count_buffer.data();
    launch_params.photon_params.photon_emitted_count = m_photon_emitted_count_buffer.data();

    // Reset photon store count
    PhotonMapStoreCount photon_store_count;
    photon_store_count.actual_count = 0;
    photon_store_count.desired_count = 0;
    m_photon_store_count_buffer.upload(&photon_store_count);
    // NOTE: Photon emitted count is not reset here, only when the subframe advances!


    m_camera->getCameraData(launch_params.image_params.camera);

    launch_params.emitters_cdf = m_emitters_cdf_buffer.view();
    launch_params.emitters = m_emitters_buffer.view();

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = m_surface_miss_index;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = m_occlusion_miss_index;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    m_launch_params_buffer.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params_buffer.getRaw(), m_launch_params_buffer.byteSize(), sbt->getSBT(m_trace_lights_raygen_index), m_photon_thread_count, 1, 1) );
    CUDA_SYNC_CHECK();

    buildPhotonMapKDTree(stream);
}

void PhotonMappingRayGenerator::launchGatherPhotons(CUstream stream)
{
    opg::BufferView<PhotonData> photon_map_buffer_view = m_photon_map_buffer.view();

    opg::BufferView<PhotonGatherData> photon_gather_buffer_view = m_photon_gather_buffer.view();
    opg::BufferView<glm::vec3> accum_photon_buffer_view = m_accum_buffer_photons.view().reinterpret<glm::vec3>();

    gatherPhotons(
        photon_map_buffer_view,
        photon_gather_buffer_view,
        accum_photon_buffer_view,
        m_photon_store_count_buffer.data(),
        m_photon_emitted_count_buffer.data(),
        m_gather_alpha
    );
    CUDA_SYNC_CHECK();
}


void PhotonMappingRayGenerator::launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer)
{
    // NOTE: We access tensors like numpy arrays.
    // 1st tensor dimension -> row -> y axis
    // 2nd tensor dimension -> column -> x axis
    uint32_t image_width = output_buffer.counts[1];
    uint32_t image_height = output_buffer.counts[0];

    // When the framebuffer resolution changes, or when the camera moves, we have to discard the previously accumulated samples and start anew
    if (m_accum_buffer_height != image_height || m_accum_buffer_width != image_width)
    {
        m_accum_buffer_width = image_width;
        m_accum_buffer_height = image_height;

        // Reallocate accum buffer on resize
        m_accum_buffer_path.alloc(m_accum_buffer_height * m_accum_buffer_width);
        // Reallocate accum buffer for photon contributions to pixels
        m_accum_buffer_photons.alloc(m_accum_buffer_height * m_accum_buffer_width);

        // Also reallocate photon gather buffer
        m_photon_gather_buffer.alloc(m_accum_buffer_height * m_accum_buffer_width);

        // Ensure that we have enough space to store the samples in the sample buffer
        m_sample_buffer.alloc(m_accum_buffer_height * m_accum_buffer_width);

        // Reset sample count
        m_accum_sample_count = 0;
    }
    else if (m_camera->getRevision() != m_camera_revision)
    {
        // Reset sample count
        m_accum_sample_count = 0;
    }
    m_camera_revision = m_camera->getRevision();

    if (m_accum_sample_count == 0)
    {
        // Reset the internal state of the photon gather buffer
        opg::BufferView<PhotonGatherData> photon_gather_buffer_view = m_photon_gather_buffer.view();
        resetPhotonGatherData(photon_gather_buffer_view, m_gather_radius*m_gather_radius);

        // Reset the number of photons emitted!
        CUDA_CHECK( cudaMemsetAsync(m_photon_emitted_count_buffer.data(), 0, m_photon_emitted_count_buffer.byteSize(), stream) );
    }


    launchTraceImage(stream);
    launchTracePhotons(stream);
    launchGatherPhotons(stream);

    opg::TensorView<glm::vec3, 2> accum_path_tensor_view = opg::make_tensor_view(m_accum_buffer_path.view().reinterpret<glm::vec3>(), image_height, image_width);
    opg::TensorView<glm::vec3, 2> accum_photon_tensor_view = opg::make_tensor_view(m_accum_buffer_photons.view().reinterpret<glm::vec3>(), image_height, image_width);
    combineOutputRadiance(output_buffer, accum_path_tensor_view, accum_photon_tensor_view);

    // Advance subframe index and accumulated sample count.
    m_subframe_index++;
    m_accum_sample_count++;

    CUDA_SYNC_CHECK();
}


static void buildKDTreeImpl(PhotonData** photons, int begin, int end, int depth, PhotonData* kd_tree, int current_root,
                  const opg::Aabb &aabb)
{
    // If we have zero photons, this is a NULL node
    if( end - begin == 0 )
    {
        kd_tree[current_root].node_type = KDNodeType::Empty;
        kd_tree[current_root].irradiance_weight = glm::vec3(0.0f);
        return;
    }

    // If we have a single photon, this is a leaf node.
    if( end - begin == 1 )
    {
        photons[begin]->node_type = KDNodeType::Leaf;
        kd_tree[current_root] = *(photons[begin]);
        return;
    }

    // If we have two photons, there is only one child.
    if( end - begin == 2 )
    {
        // Leaf node with two photons
        photons[begin]->node_type = KDNodeType::DoubleLeaf;
        kd_tree[current_root] = *(photons[begin]);
        // NOTE: Index of child nodes would be beyond photon map buffer size!
        photons[begin+1]->node_type = KDNodeType::Leaf;
        kd_tree[current_root+1] = *(photons[begin+1]);
        return;
    }

    // Choose axis to split on
    int axis = aabb.longestAxis();

    // Round up end to the next 2^n - 1
    int filled_size = end-begin;
    for (int i = 0; i < 32; ++i)
    {
        if ((1 << i) > filled_size)
        {
            filled_size &= ~(1 << (i - 1));
            break;
        }
        filled_size |= 1 << i;
    }
    int last_line_count = end - begin - filled_size;
    int median = begin + (filled_size - 1) / 2 + glm::min(last_line_count, (filled_size + 1) / 2);
    // int median = (begin+end) / 2;

    switch (axis)
    {
        case 0:
        {
            select_kth_element<PhotonData*>(photons, begin, end-1, median, [](const PhotonData* lhs, const PhotonData *rhs) -> bool {
                return lhs->position.x < rhs->position.x;
            });
            photons[median]->node_type = KDNodeType::AxisX;
            break;
        }
        case 1:
        {
            select_kth_element<PhotonData*>(photons, begin, end-1, median, [](const PhotonData* lhs, const PhotonData *rhs) -> bool {
                return lhs->position.y < rhs->position.y;
            });
            photons[median]->node_type = KDNodeType::AxisY;
            break;
        }
        case 2:
        {
            select_kth_element<PhotonData*>(photons, begin, end-1, median, [](const PhotonData* lhs, const PhotonData *rhs) -> bool {
                return lhs->position.z < rhs->position.z;
            });
            photons[median]->node_type = KDNodeType::AxisZ;
            break;
        }
    }

    opg::Aabb left_aabb = aabb;
    opg::Aabb right_aabb = aabb;

    glm::vec3 mid_point = (*photons[median]).position;
    switch( axis )
    {
        case 0:
        {
            right_aabb.m_min.x = mid_point.x;
            left_aabb.m_max.x  = mid_point.x;
            break;
        }
        case 1:
        {
            right_aabb.m_min.y = mid_point.y;
            left_aabb.m_max.y  = mid_point.y;
            break;
        }
        case 2:
        {
            right_aabb.m_min.z = mid_point.z;
            left_aabb.m_max.z  = mid_point.z;
            break;
        }
    }

    kd_tree[current_root] = *(photons[median]);
    buildKDTreeImpl(photons, begin, median, depth+1, kd_tree, 2*current_root+1, left_aabb);
    buildKDTreeImpl(photons, median+1, end, depth+1, kd_tree, 2*current_root+2, right_aabb);
}

void PhotonMappingRayGenerator::buildPhotonMapKDTree(CUstream stream)
{
    PhotonMapStoreCount photon_store_count;// = m_photon_map_buffer.size();
    m_photon_store_count_buffer.download(&photon_store_count);
    uint32_t valid_photon_count = photon_store_count.actual_count;

    // std::cout << photon_store_count.actual_count << " / " << photon_store_count.desired_count << std::endl;

    std::vector<PhotonData> unordered_photons;
    unordered_photons.resize(valid_photon_count);
    m_photon_map_buffer.downloadSub(unordered_photons.data(), valid_photon_count);

    std::vector<PhotonData> kd_tree_photons;
    kd_tree_photons.resize(valid_photon_count);

    opg::Aabb photon_map_aabb;

    std::vector<PhotonData*> photon_ptrs;
    photon_ptrs.resize(valid_photon_count);
    for (int i = 0; i < valid_photon_count; ++i)
    {
        photon_ptrs[i] = &unordered_photons[i];
        // Compute the bounds of the photons
        photon_map_aabb.include(unordered_photons[i].position);
    }

    buildKDTreeImpl(photon_ptrs.data(), 0, valid_photon_count, 0, kd_tree_photons.data(), 0, photon_map_aabb);

    m_photon_map_buffer.uploadSub(kd_tree_photons.data(), valid_photon_count);
}


namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(PhotonMappingRayGenerator, "raygen.photon");

} // end namespace opg
