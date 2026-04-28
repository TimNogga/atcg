#include "opg/opg.h"

#include "opg/memory/devicebuffer.h"
#include "opg/memory/outputbuffer.h"

#include "opg/glmwrapper.h"

#include "opg/gui/interactivescenerenderer.h"
#include "opg/gui/filescenerenderer.h"

#include "opg/scene/scene.h"
#include "opg/scene/components/shapeinstance.h"
#include "opg/scene/components/buffercomponent.h"
#include "opg/scene/components/bufferviewcomponent.h"

#include "raycastingraygenerator.h"
#include "sphere.h"
#include "cylinder.h"
#include "mesh.h"

#include <GLFW/glfw3.h>

void createSceneGeometry(opg::Scene *scene)
{
    /* Implement:
     * - Create an additional scene object that is an ellipsoid with
     *      - radii (0.25, 0.75, 0.25) along its x-, y- and z-axis
     *      - rotated by PI/4 around the z-axis
     *      - translated to (0, 1, 0)
     * - Create a cylinder scene object with
     *      - with an axis-vector from center to tip of (0.6, 0.6, 0.6)
     *      - radius 0.3
     *      - transpated to (0, -1, 0)
     */

    /// Add an ellipsoid with radii (0.25, 0.75, 0.25) rotated by pi/4 around the z axis translated to (0, 1, 0)
    {
        //
    }

    /// Add a cylinder at (0, -1, 0) with an axis-vector from center to tip of (0.6, 0.6, 0.6) and radius 0.3
    {
        //
    }

    /// Add a sphere translated to ( 1, 0, 0)
    {
        SphereShape *sphere = scene->createSceneComponent<SphereShape>(opg::Properties());

        glm::mat4 to_world = glm::translate(glm::vec3(1, 0, 0));
        to_world = glm::scale(to_world, glm::vec3(0.5f));

        opg::Properties props;
        props.setComponent("shape", sphere);
        props.setMatrix("to_world", to_world);
        auto ellipsoidInstance = scene->createSceneComponent<opg::ShapeInstance>(props);
    }

    // Add a cube scaled by 0.5 translated to (-1, 0, 0)
    {
        std::vector<glm::vec3> positions = {
            // +x
            glm::vec3( 1,-1,-1), glm::vec3( 1,-1, 1), glm::vec3( 1, 1, 1), glm::vec3( 1, 1,-1),
            // +y
            glm::vec3(-1, 1,-1), glm::vec3(-1, 1, 1), glm::vec3( 1, 1, 1), glm::vec3( 1, 1,-1),
            // +z
            glm::vec3(-1,-1, 1), glm::vec3(-1, 1, 1), glm::vec3( 1, 1, 1), glm::vec3( 1,-1, 1),
            // -x
            glm::vec3(-1,-1,-1), glm::vec3(-1,-1, 1), glm::vec3(-1, 1, 1), glm::vec3(-1, 1,-1),
            // -y
            glm::vec3(-1,-1,-1), glm::vec3(-1,-1, 1), glm::vec3( 1,-1, 1), glm::vec3( 1,-1,-1),
            // -z
            glm::vec3(-1,-1,-1), glm::vec3(-1, 1,-1), glm::vec3( 1, 1,-1), glm::vec3( 1,-1,-1),
        };
        opg::DeviceBuffer<glm::vec3> d_positions;
        d_positions.alloc(positions.size());
        d_positions.upload(positions.data());
        opg::BufferComponent *positions_component = scene->createSceneComponent<opg::BufferComponent>(std::move(d_positions), opg::Properties());

        opg::BufferViewComponent *positions_view_component = scene->createSceneComponent<opg::BufferViewComponent>(opg::Properties()
            .setComponent("data", positions_component)
            .setInt("offset", 0)
            .setInt("element_size", sizeof(float))
            .setInt("stride", sizeof(glm::vec3))
            .setInt("count", positions.size())
        );

        std::vector<glm::vec3> normals = {
            // +x
            glm::vec3( 1, 0, 0), glm::vec3( 1, 0, 0), glm::vec3( 1, 0, 0), glm::vec3( 1, 0, 0),
            // +y
            glm::vec3( 0, 1, 0), glm::vec3( 0, 1, 0), glm::vec3( 0, 1, 0), glm::vec3( 0, 1, 0),
            // +z
            glm::vec3( 0, 0, 1), glm::vec3( 0, 0, 1), glm::vec3( 0, 0, 1), glm::vec3( 0, 0, 1),
            // -x
            glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
            // -y
            glm::vec3( 0,-1, 0), glm::vec3( 0,-1, 0), glm::vec3( 0,-1, 0), glm::vec3( 0,-1, 0),
            // -z
            glm::vec3( 0, 0,-1), glm::vec3( 0, 0,-1), glm::vec3( 0, 0,-1), glm::vec3( 0, 0,-1),
        };
        opg::DeviceBuffer<glm::vec3> d_normals;
        d_normals.alloc(normals.size());
        d_normals.upload(normals.data());
        opg::BufferComponent *normals_component = scene->createSceneComponent<opg::BufferComponent>(std::move(d_normals), opg::Properties());

        opg::BufferViewComponent *normals_view_component = scene->createSceneComponent<opg::BufferViewComponent>(opg::Properties()
            .setComponent("data", normals_component)
            .setInt("offset", 0)
            .setInt("element_size", sizeof(float))
            .setInt("stride", sizeof(glm::vec3))
            .setInt("count", normals.size())
        );

        std::vector<glm::vec2> uvs = {
            // +x
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
            // +y
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
            // +z
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
            // -x
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
            // -y
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
            // -z
            glm::vec2( 0, 0), glm::vec2( 0, 1), glm::vec2( 1, 1), glm::vec2( 1, 0),
        };
        opg::DeviceBuffer<glm::vec2> d_uvs;
        d_uvs.alloc(uvs.size());
        d_uvs.upload(uvs.data());
        opg::BufferComponent *uvs_component = scene->createSceneComponent<opg::BufferComponent>(std::move(d_uvs), opg::Properties());

        opg::BufferViewComponent *uvs_view_component = scene->createSceneComponent<opg::BufferViewComponent>(opg::Properties()
            .setComponent("data", uvs_component)
            .setInt("offset", 0)
            .setInt("element_size", sizeof(float))
            .setInt("stride", sizeof(glm::vec2))
            .setInt("count", uvs.size())
        );

        std::vector<glm::u16vec3> indices = {
            // +x
            glm::u16vec3(0, 2, 1), glm::u16vec3(0, 3, 2),
            // +y
            glm::u16vec3(4, 5, 6), glm::u16vec3(4, 6, 7),
            // +z
            glm::u16vec3(8, 10, 9), glm::u16vec3(8, 11, 10),
            // -x
            glm::u16vec3(12, 13, 14), glm::u16vec3(12, 14, 15),
            // -y
            glm::u16vec3(16, 18, 17), glm::u16vec3(16, 19, 18),
            // -z
            glm::u16vec3(20, 21, 22), glm::u16vec3(20, 22, 23)
        };
        opg::DeviceBuffer<glm::u16vec3> d_indices;
        d_indices.alloc(indices.size());
        d_indices.upload(indices.data());
        opg::BufferComponent *indices_component = scene->createSceneComponent<opg::BufferComponent>(std::move(d_indices), opg::Properties());

        opg::BufferViewComponent *indices_view_component = scene->createSceneComponent<opg::BufferViewComponent>(opg::Properties()
            .setComponent("data", indices_component)
            .setInt("offset", 0)
            .setInt("element_size", sizeof(glm::u16vec3))
            .setInt("stride", sizeof(glm::u16vec3))
            .setInt("count", indices.size())
        );

        MeshShape *mesh = scene->createSceneComponent<MeshShape>(opg::Properties()
            .setComponent("indices", indices_view_component)
            .setComponent("positions", positions_view_component)
            .setComponent("normals", normals_view_component)
            .setComponent("uvs", uvs_view_component)
        );

        glm::mat4 to_world = glm::mat4(1);
        to_world = glm::translate(to_world, glm::vec3(-1, 0, 0));
        to_world = glm::scale(to_world, glm::vec3(0.5f));

        opg::ShapeInstance *meshInstance = scene->createSceneComponent<opg::ShapeInstance>(opg::Properties()
            .setComponent("shape", mesh)
            .setMatrix("to_world", to_world)
        );
    }
}


void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
              << "Options:\n"
              << "  -h | --help                                 Print this usage message\n"
              << "  -o | --output <output.png>                  Output filename (optional)\n"
              << "                                                  No GUI will be started if present\n"
              << "  -r | --resolution <width> <height>          Output image width and height\n"
              << std::endl;

    exit(1);
}

int main(int argc, char **argv)
{
    int output_width = 512;
    int output_height = 512;
    std::string output_filename;

    // parse arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg( argv[i] );
        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "-o" || arg == "--output" ) && i + 1 < argc )
        {
            output_filename = argv[++i];
        }
        else if( ( arg == "-r" || arg == "--resolution" ) && i + 2 < argc )
        {
            output_width = std::atoi(argv[++i]);
            output_height = std::atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Bad option: '" << arg << "'" << std::endl;
            printUsageAndExit( argv[0] );
        }
    }

    // Initialize CUDA
    auto ctx = opg::createOptixContext();

    // Create a scene
    opg::Scene scene(ctx);
    createSceneGeometry(&scene);

    // Add a camera to the scene that can be controlled by the user
    glm::vec3 eye = glm::vec3(0, 0, 5);
    glm::vec3 up = glm::vec3(0, 1, 0);
    glm::vec3 center = glm::vec3(0, 0, 0);
    // Since glm is designed to be used with OpenGL rasterizer frameworks, the glm::lookAt function
    // produces a transformation from the world into the camera coordinate system
    // In our ray tracing application, we define rays in the camera coordinate system and transform
    // them into the world coordinate system.
    glm::mat4 world_to_camera = glm::lookAt(eye, center, up);
    glm::mat4 camera_to_world = glm::inverse(world_to_camera);
    opg::Camera *camera = scene.createSceneComponent<opg::Camera>(opg::Properties()
        .setFloat("aspect_ratio", static_cast<float>(output_height) / static_cast<float>(output_width))
        .setFloat("fov_y", 45.0f * M_PIf/180.0f)
        .setMatrix("to_world", camera_to_world)
        .setFloat("lookat_distance", glm::length(eye-center))
    );

    // Add a ray generator component that controlls the ray-tracing logic, i.e. casts rays and writes colors to the output buffer!
    RayCastingRayGenerator *raygen = scene.createSceneComponent<RayCastingRayGenerator>(opg::Properties());
    raygen->setCamera(camera);

    scene.finalize();


    if (output_filename.empty())
    {
        opg::InteractiveSceneRenderer renderer(&scene, output_width, output_height, OPG_TARGET_NAME);
        renderer.run();
    }
    else
    {
        opg::FileSceneRenderer renderer(&scene, output_width, output_height);
        renderer.run(output_filename, 1);
    }

    return 0;
}
