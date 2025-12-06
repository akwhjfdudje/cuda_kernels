#pragma once

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
    float erosionRate);
