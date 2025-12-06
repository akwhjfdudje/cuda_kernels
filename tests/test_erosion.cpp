#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "render/render.cuh"
#include <cuda_runtime.h>

// Host helper to initialize height, water, and sediment maps
void initMaps(std::vector<float>& heightmap, std::vector<float>& watermap, std::vector<float>& sedimentmap, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>((i % 10) - 5); // initial terrain heights
        watermap[i] = 0.0f;                                // no initial water
        sedimentmap[i] = 0.0f;                             // no initial sediment
    }
}

// Check that height values are finite and within a reasonable bound
void checkHeightmap(const std::vector<float>& heightmap, float bound = 1000.0f) {
    for (float val : heightmap) {
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_GE(val, -bound);
        EXPECT_LE(val, bound);
    }
}

// Small grid test
TEST(ErosionKernel, SmallGrid) {
    int width = 8;
    int height = 8;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height);
    std::vector<float> sedimentmap(width * height);

    initMaps(heightmap, watermap, sedimentmap, width, height);

    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));

    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Run one erosion step
    erodeHeightmap(d_h, d_w, d_s, width, height,
                   0.1f,   // timestep
                   0.01f,  // rainAmount
                   0.001f, // evapRate
                   1.0f,   // capacity
                   0.1f,   // depositRate
                   0.1f    // erosionRate
    );

    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    checkHeightmap(heightmap);

    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

// Moderate grid test
TEST(ErosionKernel, ModerateGrid) {
    int width = 64;
    int height = 64;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height);
    std::vector<float> sedimentmap(width * height);

    initMaps(heightmap, watermap, sedimentmap, width, height);

    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));

    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Run 5 erosion steps
    for (int i = 0; i < 5; i++) {
        erodeHeightmap(d_h, d_w, d_s, width, height,
                       0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    }

    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    checkHeightmap(heightmap);

    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

// Large grid test
TEST(ErosionKernel, LargeGrid) {
    int width = 512;
    int height = 512;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height);
    std::vector<float> sedimentmap(width * height);

    initMaps(heightmap, watermap, sedimentmap, width, height);

    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));

    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Run 10 erosion steps
    for (int i = 0; i < 10; i++) {
        erodeHeightmap(d_h, d_w, d_s, width, height,
                       0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    }

    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    checkHeightmap(heightmap);

    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

