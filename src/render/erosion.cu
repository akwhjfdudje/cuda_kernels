/**
 * @file render/erosion.cu
 * @brief CUDA kernel for simple hydraulic erosion on a heightmap.
 */

#include <cuda_runtime.h>
#include <math.h>
#include "render/render.cuh"

#define THREADS_PER_BLOCK 256

/**
 * @brief CUDA kernel performing one step of hydraulic erosion.
 *
 * Each thread handles one heightmap cell and updates its height
 * based on water flow and sediment transport with neighbors.
 *
 * @param height_in Input heightmap buffer (width*height floats)
 * @param water_in Input water depth per cell (same size as heightmap)
 * @param sed_in Input sediment carried by water (same size)
 * @param height_out Output heightmap buffer (width*height floats)
 * @param water_out Output water buffer
 * @param sed_out Output sediment buffer
 * @param width Width of heightmap
 * @param height Height of heightmap
 * @param timeStep Simulation step size
 * @param rainAmount Amount of water added per cell per step
 * @param evapRate Evaporation rate (fraction)
 * @param capacity Sediment capacity factor
 * @param depositRate Rate of sediment deposition
 * @param erosionRate Rate of terrain erosion
 */
__global__ void erosionKernel(
    float* height_in, float* water_in, float* sed_in,
    float* height_out, float* water_out, float* sed_out,
    int width, int height,
    float timeStep, float rainAmount,
    float evapRate, float capacity,
    float depositRate, float erosionRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;

    float h = height_in[idx];
    float w = water_in[idx];
    float s = sed_in[idx];

    // 1. Add rain
    w += rainAmount * timeStep;

    // Neighbor offsets
    const int nx[4] = {0, 1, 0, -1};
    const int ny[4] = {-1, 0, 1, 0};

    float totalDiff = 0.f;
    float diffs[4];

    float thisTotal = h + w;

    // Compute height differences
    for (int i = 0; i < 4; i++) {
        int xx = x + nx[i];
        int yy = y + ny[i];
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) {
            diffs[i] = 0;
            continue;
        }
        int nIdx = yy * width + xx;

        float diff = thisTotal - (height_in[nIdx] + water_in[nIdx]);
        diffs[i] = (diff > 0 ? diff : 0);
        totalDiff += diffs[i];
    }

    // Flow factor: DON'T MOVE ALL WATER IN ONE ITERATION
    const float FLOW_RATE = 0.25f; // 0.01–0.25 recommended

    float w_outflow = 0.f;
    float s_outflow = 0.f;

    // Distribute flow
    for (int i = 0; i < 4; i++) {
        if (diffs[i] == 0 || totalDiff == 0) continue;

        float fraction = diffs[i] / totalDiff;
        float flow = w * fraction * FLOW_RATE;

        w_outflow += flow;
        s_outflow += (flow / fmaxf(w, 1e-6f)) * s;
    }

    float w_new = w - w_outflow;
    float s_new = s - s_outflow;

    // Erosion/deposition
    float sedCap = capacity * w_new;

    if (s_new > sedCap) {
        float dep = (s_new - sedCap) * depositRate * timeStep;
        s_new -= dep;
        h += dep;
    } else {
        float erode = (sedCap - s_new) * erosionRate * timeStep;
        s_new += erode;
        h -= erode;
    }

    // Evaporate
    w_new *= (1.f - evapRate * timeStep);

    // Write results to output buffers
    height_out[idx] = h;
    water_out[idx]  = w_new;
    sed_out[idx]    = s_new;
}


/**
 * @brief Host launcher for erosion kernel.
 */
extern "C" CUDA_KERNELS_API void erodeHeightmap(
    float* heightmap,
    float* watermap,
    float* sedimentmap,
    int width,
    int height,
    float timeStep,
    float rainAmount,
    float evapRate,
    float capacity,
    float depositRate,
    float erosionRate)
{
    int total = width * height;
    float *h2, *w2, *s2;

    cudaMalloc(&h2, total*sizeof(float));
    cudaMalloc(&w2, total*sizeof(float));
    cudaMalloc(&s2, total*sizeof(float));

    dim3 threads(256);
    dim3 blocks((total + 255) / 256);

    erosionKernel<<<blocks,threads>>>(
        heightmap, watermap, sedimentmap,
        h2, w2, s2,
        width, height,
        timeStep, rainAmount,
        evapRate, capacity,
        depositRate, erosionRate
    );

    cudaMemcpy(heightmap, h2, total*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(watermap,  w2, total*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sedimentmap, s2, total*sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(h2); cudaFree(w2); cudaFree(s2);

    cudaDeviceSynchronize();
}
