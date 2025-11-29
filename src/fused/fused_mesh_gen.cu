/**
 * @file fused/fused_mesh_gen.cu
 * @brief Generate mesh (vertices, normals, texcoords) from a device heightmap.
 *        All pointers are raw device pointers.
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
    float scaleX,
    float scaleY,
    float heightScale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float h = heightmap[idx];

    // vertex: note coordinate system: Y is up
    float vx = x * scaleX;
    float vy = h * heightScale;
    float vz = y * scaleY;
    vertices[idx] = make_float3(vx, vy, vz);

    // texcoord (0..1)
    texcoords[idx] = make_float2(
        (width > 1) ? (float)x / (float)(width - 1) : 0.0f,
        (height > 1) ? (float)y / (float)(height - 1) : 0.0f
    );

    // finite differences with clamping
    int xm = (x > 0) ? x - 1 : x;
    int xp = (x < width - 1) ? x + 1 : x;
    int ym = (y > 0) ? y - 1 : y;
    int yp = (y < height - 1) ? y + 1 : y;

    float hL = heightmap[y * width + xm];
    float hR = heightmap[y * width + xp];
    float hD = heightmap[ym * width + x];
    float hU = heightmap[yp * width + x];

    // compute gradient in world space (scale back into world units)
    // dx = (hR - hL) / (2 * scaleX) if you want slope per world unit
    float dx = (hR - hL) * 0.5f / ((scaleX != 0.0f) ? scaleX : 1.0f);
    float dz = (hU - hD) * 0.5f / ((scaleY != 0.0f) ? scaleY : 1.0f);

    float3 n;
    n.x = -dx;
    n.y = 1.0f;
    n.z = -dz;

    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-8f) {
        n.x /= len; n.y /= len; n.z /= len;
    } else {
        n = make_float3(0.0f, 1.0f, 0.0f);
    }

    normals[idx] = n;
}

extern "C" CUDA_KERNELS_API void generateMeshFromHeightmap(
    const float* d_heightmap,
    float3* d_vertices,
    float3* d_normals,
    float2* d_texcoords,
    int width,
    int height,
    float scaleX,
    float scaleY,
    float heightScale)
{
    dim3 threads(MESH_BLOCK_X, MESH_BLOCK_Y);
    dim3 blocks( (width + threads.x - 1) / threads.x,
                 (height + threads.y - 1) / threads.y );

    generateMeshKernel<<<blocks, threads>>>(
        d_heightmap,
        d_vertices,
        d_normals,
        d_texcoords,
        width,
        height,
        scaleX,
        scaleY,
        heightScale
    );

    cudaDeviceSynchronize();
}

