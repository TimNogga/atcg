#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>

#include "opg/opg.h"
#include "opg/exception.h"
#include "opg/imagedata.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"

#include "kernels.h"

void taskA()
{
    /* Task:
     * Using the GLM math library:
     * - Consider the equation $<n, x> + d = 0$ in 3d space that holds for all points on a plane, where n is a unit-length vector and d is a scalar.
     *   Determine the parameters n, d that describes the plane spanned by the given plane points.
     * - Swap the y and z components of the plane normal.
     * - Find the intersection point between the *new* plane (modified n, old d) and the ray.
     *    - Note: the ray parameters are given as CUDA-builtin float3 vectors.
     *    - Convert the ray parameters to glm::vec3 using the cuda2glm() function.
     *    - Implement the ray-plane intersection.
     *    - Convert the intersection point back to a CUDA-builtin float3 vector using the glm2cuda() function.
     */

    std::cout << "taskA output:" << std::endl;

    // Plane equiation: dot(n, x) + d = 0
    glm::vec3 plane_point_1 = glm::vec3(3, 0, 2);
    glm::vec3 plane_point_2 = glm::vec3(4, 3, 2);
    glm::vec3 plane_point_3 = glm::vec3(1, 2, 4);

    // TODO implement
    glm::vec3 plane_normal = glm::normalize(glm::cross(plane_point_2 - plane_point_1, plane_point_3 - plane_point_1));
    float     plane_d = -glm::dot(plane_normal, plane_point_1);

    //

    std::cout << "\tplane normal:             " << plane_normal << ", plane offset: " << plane_d << std::endl;

    // TODO swap plane_normal y,z components
    std::swap(plane_normal.y, plane_normal.z);
    //

    std::cout << "\tswizzled plane normal:    " << plane_normal << std::endl;

    // Ray equation: x = origin + t * dir
    float3 ray_origin_cudafloat3 = make_float3(1, 4, 2);
    float3 ray_dir_cudafloat3 = make_float3(2, 4, 3);

    float3 intersection_cudafloat3;
    // TOOD:
    // - convert ray parameters to GLM vectors using cuda2glm() function.
    // - implement ray-plane intersection.
    // - convert intersection point back to float3 type using glm2cuda() function.
    glm::vec3 ray_origin = cuda2glm(ray_origin_cudafloat3);
    glm::vec3 ray_dir = cuda2glm(ray_dir_cudafloat3);

    float t = -(plane_d + glm::dot(plane_normal, ray_origin)) / glm::dot(plane_normal, ray_dir);
    glm::vec3 intersection = ray_origin + t * ray_dir;

    intersection_cudafloat3 = glm2cuda(intersection);
    //

    std::cout << "\tray-plane intersection at (" << intersection_cudafloat3.x << " " << intersection_cudafloat3.y << " " << intersection_cudafloat3.z << ")" << std::endl;
}

void taskB()
{
    /* Task:
     * - Generate an array containing integer numbers 1..1e7 and copy it to the GPU.
     * - Use a CUDA kernel to multiply each number by a constant in parallel.
     * - Copy the result back to the CPU host memory.
     */

    // Use dataArray as your host memory
    std::vector<int> dataArray;

    // Use d_dataArray as a pointer to device memory
    int *d_dataArray;

    int n = 1e7;
    dataArray.resize(n);

    std::iota(dataArray.begin(), dataArray.end(), 1);

    cudaMallocManaged(&d_dataArray, n * sizeof(int));

    std::memcpy(d_dataArray, dataArray.data(), n * sizeof(int));

    cudaMemPrefetchAsync(d_dataArray, n*sizeof(int), 0, 0);
    runMultiply(n, d_dataArray, 7);

    std::memcpy(dataArray.data(), d_dataArray, n * sizeof(int));

    cudaFree(d_dataArray);

    //

    std::cout << "taskB output:" << std::endl;
    std::cout << "\tfirst 10 entries:";
    for (int i = 0; i < std::min<int>(dataArray.size(), 10); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
    std::cout << "\tlast 10 entries:";
    for (int i = std::max<int>(dataArray.size()-10, 0); i < dataArray.size(); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
}

void taskC()
{
    /* Task:
     * - Apply the separable Sobel filter to an image.
     *   Use two kernel executions to compute a single 2D convolution.
     * - Output G = sqrt(G_x^2 + G_y^2)
     */

    opg::ImageData imageData;
    // opg::getRootPath returns the absolute path to the root directory
    // of this framework independent of the current working directory
    std::string filename = opg::getRootPath() + "/data/exercise00_CUDAIntro/Valve.png";
    opg::readImage(filename.c_str(), imageData);
    uint32_t channelSize = opg::getImageFormatChannelSize(imageData.format);
    uint32_t channelCount = opg::getImageFormatChannelCount(imageData.format);

    // Each color channel stores a single uint8_t (or unsigned char) value per pixel.
    OPG_CHECK(channelSize == 1);

    int n = imageData.width * imageData.height;

    // move image data to gpu
    uint8_t* d_image;
    cudaMallocManaged(&d_image, n*channelCount);
    std::memcpy(d_image, imageData.data.data(), n*channelCount);

    // create gradient images on gpu
    int *G_x, *G_y, *G_x_intermediate, *G_y_intermediate;
    cudaMallocManaged(&G_x, n*channelCount*sizeof(int));
    cudaMallocManaged(&G_y, n*channelCount*sizeof(int));
    cudaMallocManaged(&G_x_intermediate, n*channelCount*sizeof(int));
    cudaMallocManaged(&G_y_intermediate, n*channelCount*sizeof(int));

    // create kernels
    std::vector<int> x_horizontal = { 1, 0, -1 };
    std::vector<int> x_vertical = { 1, 2, 1 };
    std::vector<int> y_horizontal = { 1, 2, 1 };
    std::vector<int> y_vertical = { 1, 0, -1 };
    int *d_x_horizontal, *d_x_vertical, *d_y_horizontal, *d_y_vertical;
    cudaMallocManaged(&d_x_horizontal, 3*sizeof(int)); std::memcpy(d_x_horizontal, x_horizontal.data(), 3*sizeof(int));
    cudaMallocManaged(&d_x_vertical, 3*sizeof(int)); std::memcpy(d_x_vertical, x_vertical.data(), 3*sizeof(int));
    cudaMallocManaged(&d_y_horizontal, 3*sizeof(int)); std::memcpy(d_y_horizontal, y_horizontal.data(), 3*sizeof(int));
    cudaMallocManaged(&d_y_vertical, 3*sizeof(int)); std::memcpy(d_y_vertical, y_vertical.data(), 3*sizeof(int));

    // horizontal pass
    runHorizontalPass(n, imageData.width, imageData.height, channelCount, d_image, G_x_intermediate, d_x_horizontal);
    runHorizontalPass(n, imageData.width, imageData.height, channelCount, d_image, G_y_intermediate, d_y_horizontal);

    CUDA_SYNC_CHECK();

    // vertical pass
    runVerticalPass(n, imageData.width, imageData.height, channelCount, G_x_intermediate, G_x, d_x_vertical);
    runVerticalPass(n, imageData.width, imageData.height, channelCount, G_y_intermediate, G_y, d_y_vertical);

    CUDA_SYNC_CHECK();

    // compute magnitude
    runComputeMagnitude(n, channelCount, d_image, G_x, G_y);

    CUDA_SYNC_CHECK();

    std::memcpy(imageData.data.data(), d_image, n*channelCount);

    cudaFree(d_image);
    cudaFree(G_x);
    cudaFree(G_y);
    cudaFree(G_x_intermediate);
    cudaFree(G_y_intermediate);
    cudaFree(d_x_horizontal);
    cudaFree(d_x_vertical);
    cudaFree(d_y_horizontal);
    cudaFree(d_y_vertical);

    //

    // Write your solution back into the imageData.data array such that we can safe it as an image again
    opg::writeImagePNG("taskC_output.png", imageData);
}

void taskD()
{
    /* Task:
     * - Create an array of uniformly distributed (pseudo) random floats in [0, 1] on the GPU.
     *   You can use tea<4>(a, b) to initialize a random number generator, and rnd(seed) to generate a pseudo random number on the GPU. (See sutil/cuda/random.h)
     * - Count all entries that are greater than 0.5 in parallel.
     */

    float threshold = 0.5f;
    int totalCount = 10000000;
    int aboveThresholdCount = 0; // compute this value

    float* d_data;
    cudaMallocManaged(&d_data, totalCount * sizeof(float));

    runRandom(totalCount, d_data);
    CUDA_SYNC_CHECK();

    runCount(totalCount, d_data, threshold, &aboveThresholdCount);
    CUDA_SYNC_CHECK();

    cudaFree(d_data);

    //

    std::cout << "taskD output:" << std::endl;
    std::cout << "\t" << aboveThresholdCount << "/" << totalCount << " values are greater than " << threshold << std::endl;
}

void taskE()
{
    /* Task:
     * - Implement a matrix multiplication in a CUDA kernel.
     *   Store your matrices in row major order (https://en.wikipedia.org/wiki/Row-_and_column-major_order).
     */

    int lhsRows = 4;
    int lhsCols = 10;
    std::vector<float> lhs = { // 4x10
        97, 95, 80, 31, 31, 72,  1,  2, 88, 93,
        46, 58, 54, 94, 84, 59, 75,  4, 78, 62,
        44, 92, 14, 83, 82, 47, 78, 88, 28, 12,
        22, 96, 61, 93, 95, 77, 27, 35, 58, 53
    };
    int rhsRows = 10;
    int rhsCols = 3;
    std::vector<float> rhs = { // 10x3
        79, 64, 67,
        67, 14, 43,
        68, 17, 43,
        75,  4,  3,
        94, 81, 83,
         1, 56, 18,
        52, 59,  0,
        88, 89, 96,
        93, 66, 76,
        38, 33, 12
    };
    int outputRows = 4;
    int outputCols = 3;
    std::vector<float> output;
    output.resize(outputRows * outputCols);
    // Expected output:
    // 36725, 24679, 25982,
    // 40059, 27133, 23270,
    // 39432, 28626, 26127,
    // 40192, 26453, 26179

    float *d_out, *d_lhs, *d_rhs;
    cudaMallocManaged(&d_out, output.size()*sizeof(float));
    cudaMallocManaged(&d_lhs, lhs.size()*sizeof(float));
    cudaMallocManaged(&d_rhs, rhs.size()*sizeof(float));

    std::memcpy(d_lhs, lhs.data(), lhs.size()*sizeof(float));
    std::memcpy(d_rhs, rhs.data(), rhs.size()*sizeof(float));

    runMatrixMultiply(lhsRows, lhsCols, d_lhs, rhsRows, rhsCols, d_rhs, d_out);
    CUDA_SYNC_CHECK();

    std::memcpy(output.data(), d_out, output.size()*sizeof(float));

    cudaFree(d_out);
    cudaFree(d_lhs);
    cudaFree(d_rhs);

    //

    std::cout << "taskE output:" << std::endl;
    for (int r = 0; r < outputRows; ++r)
    {
        std::cout << "\t";
        for (int c = 0; c < outputCols; ++c)
        {
            std::cout << " " << output[r * outputCols + c];
        }
        std::cout << std::endl;
    }
}

int main()
{
    taskA();
    taskB();
    taskC();
    taskD();
    taskE();
}
