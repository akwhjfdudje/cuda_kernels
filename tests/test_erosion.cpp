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

TEST(ErosionKernel, SinglePixel) {
    int width = 1, height = 1;
    std::vector<float> heightmap = {5.0f};
    std::vector<float> watermap = {0.0f};
    std::vector<float> sedimentmap = {0.0f};
    
    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_s, sizeof(float));
    
    cudaMemcpy(d_h, heightmap.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), sizeof(float), cudaMemcpyHostToDevice);
    
    erodeHeightmap(d_h, d_w, d_s, width, height, 0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    
    cudaMemcpy(heightmap.data(), d_h, sizeof(float), cudaMemcpyDeviceToHost);
    
    checkHeightmap(heightmap);
    
    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

TEST(ErosionKernel, FlatTerrain) {
    int width = 8, height = 8;
    std::vector<float> heightmap(width * height, 10.0f);
    std::vector<float> watermap(width * height, 0.0f);
    std::vector<float> sedimentmap(width * height, 0.0f);
    
    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));
    
    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    erodeHeightmap(d_h, d_w, d_s, width, height, 0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    
    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    checkHeightmap(heightmap);
    
    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

TEST(ErosionKernel, SteepSlope) {
    int width = 16, height = 16;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height, 0.0f);
    std::vector<float> sedimentmap(width * height, 0.0f);
    
    // Create a steep slope
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            heightmap[y * width + x] = static_cast<float>(x + y);
        }
    }
    
    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));
    
    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    erodeHeightmap(d_h, d_w, d_s, width, height, 0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    
    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    checkHeightmap(heightmap);
    
    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

TEST(ErosionKernel, MultipleIterations) {
    int width = 32, height = 32;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height, 0.0f);
    std::vector<float> sedimentmap(width * height, 0.0f);
    
    initMaps(heightmap, watermap, sedimentmap, width, height);
    
    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));
    
    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run many iterations
    for (int i = 0; i < 20; ++i) {
        erodeHeightmap(d_h, d_w, d_s, width, height, 0.1f, 0.01f, 0.001f, 1.0f, 0.1f, 0.1f);
    }
    
    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    checkHeightmap(heightmap);
    
    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

TEST(ErosionKernel, DifferentParameters) {
    int width = 16, height = 16;
    std::vector<float> heightmap(width * height);
    std::vector<float> watermap(width * height, 0.0f);
    std::vector<float> sedimentmap(width * height, 0.0f);
    
    initMaps(heightmap, watermap, sedimentmap, width, height);
    
    float *d_h, *d_w, *d_s;
    cudaMalloc(&d_h, width * height * sizeof(float));
    cudaMalloc(&d_w, width * height * sizeof(float));
    cudaMalloc(&d_s, width * height * sizeof(float));
    
    cudaMemcpy(d_h, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, watermap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, sedimentmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test with different parameter values
    erodeHeightmap(d_h, d_w, d_s, width, height, 0.05f, 0.02f, 0.002f, 2.0f, 0.2f, 0.2f);
    
    cudaMemcpy(heightmap.data(), d_h, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    checkHeightmap(heightmap);
    
    cudaFree(d_h);
    cudaFree(d_w);
    cudaFree(d_s);
}

