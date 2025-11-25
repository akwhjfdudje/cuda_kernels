/**
 * @file fused/fused_mesh_gen.cu
 * @brief Generate vertex positions, normals, and texcoords from a heightmap
 *        directly into CUDA-memory.
 *
 * Kernel layout:
 *   - Each thread handles one heightmap texel (x,y).
 *   - Writes out vertices[idx] = { x*scale, height, y*scale }
 *   - Computes normal via finite differences into normals[idx]
 *   - Writes texcoords[idx] = { x/width, y/height }
 * @note Currently, there are no plans to add a test suite 
 *       for this kernel, as it relies on the GPU completely.
 */

#include <cuda_runtime.h>
#include <math.h>
#include "fused/fused.cuh"

#define MESH_BLOCK_X 16
#define MESH_BLOCK_Y 16

__global__ void generateMeshKernel(
    const float* __restrict__ heightmap,
    float3* __restrict__ vertices,
    float3* __restrict__ normals,
    float2* __restrict__ texcoords,
    int width,
    int height,
    float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // sample height
    float h = heightmap[idx];

    // vertex position: x, y (height), z
    float vx = x * scale;
    float vy = h * scale * 30.0f;
    float vz = y * scale;
    vertices[idx] = make_float3(vx, vy, vz);

    // texcoord
    texcoords[idx] = make_float2((float)x / (float)(width - 1), (float)y / (float)(height - 1));

    // finite differences for normal (clamp at boundaries)
    int xm = (x > 0) ? (x - 1) : x;
    int xp = (x < width - 1) ? (x + 1) : x;
    int ym = (y > 0) ? (y - 1) : y;
    int yp = (y < height - 1) ? (y + 1) : y;

    float hL = heightmap[y * width + xm];
    float hR = heightmap[y * width + xp];
    float hD = heightmap[ym * width + x];
    float hU = heightmap[yp * width + x];

    // compute gradient; scale factor influences "steepness"
    float dx = (hR - hL) * 0.5f;
    float dz = (hU - hD) * 0.5f;

    // Build normal (note coordinate convention: up is +Y)
    float3 n;
    n.x = -dx;
    n.y = 1.0f;
    n.z = -dz;

    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-8f) {
        n.x /= len;
        n.y /= len;
        n.z /= len;
    } else {
        n.x = 0.0f; n.y = 1.0f; n.z = 0.0f;
    }

    normals[idx] = n;
}

/**
 * @brief Host wrapper that maps three GL VBOs (vertices, normals, texcoords),
 *        gets their device pointers, and launches the mesh generator kernel.
 *
 * @param heightmap Device pointer to heightmap (width*height floats)
 * @param width width
 * @param height height
 * @param scale world-space scale for x/z
 * @param vboVerticesRes cudaGraphicsResource for vertex VBO (must be registered)
 * @param vboNormalsRes  cudaGraphicsResource for normal VBO
 * @param vboTexcoordRes cudaGraphicsResource for texcoord VBO
 */
extern "C" CUDA_KERNELS_API void generateMeshToVBOs(
    const float* heightmap,
    int width,
    int height,
    float scale,
    cudaGraphicsResource_t vboVerticesRes,
    cudaGraphicsResource_t vboNormalsRes,
    cudaGraphicsResource_t vboTexcoordRes)
{
    cudaGraphicsResource_t resources[3] = { vboVerticesRes, vboNormalsRes, vboTexcoordRes };
    cudaGraphicsMapResources(3, resources, 0);

    float3* d_vertices = nullptr;
    size_t vbSize = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &vbSize, vboVerticesRes);

    float3* d_normals = nullptr;
    size_t nbSize = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_normals, &nbSize, vboNormalsRes);

    float2* d_texcoords = nullptr;
    size_t tbSize = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_texcoords, &tbSize, vboTexcoordRes);

    dim3 threads(MESH_BLOCK_X, MESH_BLOCK_Y);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    generateMeshKernel<<<blocks, threads>>>(
        heightmap, d_vertices, d_normals, d_texcoords, width, height, scale
    );

    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(3, resources, 0);
}
