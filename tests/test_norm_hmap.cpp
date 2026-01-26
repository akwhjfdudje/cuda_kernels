#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "render/render.cuh"
#include <cuda_runtime.h>

// Host helper to compute min/max on GPU output
void computeMinMax(const std::vector<float>& data, float& min_val, float& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
}

// Small grid test
TEST(NormalizeHeightmapKernel, SmallGrid) {
    int width = 8;
    int height = 8;
    std::vector<float> heightmap(width * height);

    // Fill with arbitrary values
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>(i % 5 - 2); // values between -2 and 2
    }

    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Compute actual min/max
    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);

    // Normalize
    normalizeHeightmap(d_hmap, width, height, min_val, max_val);

    // Copy back
    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val,  1.0f + 1e-5f);
    }

    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, ModerateGrid) {
    int width = 64;
    int height = 64;
    std::vector<float> heightmap(width * height);

    // Fill with random-ish pattern
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>((i * 13 % 101) - 50); // values between -50 and 50
    }

    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);

    normalizeHeightmap(d_hmap, width, height, min_val, max_val);

    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val,  1.0f + 1e-5f);
    }

    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, LargeGrid) {
    int width = 1024;
    int height = 1024;
    std::vector<float> heightmap(width * height);

    // Fill with random-ish pattern
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>((i * 13 % 101) - 50); // values between -50 and 50
    }

    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);

    normalizeHeightmap(d_hmap, width, height, min_val, max_val);

    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val,  1.0f + 1e-5f);
    }

    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, SinglePixel) {
    int width = 1, height = 1;
    std::vector<float> heightmap = {5.0f};
    
    float* d_hmap;
    cudaMalloc(&d_hmap, sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), sizeof(float), cudaMemcpyHostToDevice);
    
    // Use different min/max to avoid division by zero
    // Normalize from [0, 10] to [-1, 1]
    normalizeHeightmap(d_hmap, width, height, 0.0f, 10.0f);
    
    cudaMemcpy(heightmap.data(), d_hmap, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Value 5.0 in range [0, 10] should map to 0.0 in [-1, 1]
    EXPECT_NEAR(heightmap[0], 0.0f, 1e-5f);
    EXPECT_TRUE(std::isfinite(heightmap[0]));
    
    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, ConstantValues) {
    int width = 8, height = 8;
    std::vector<float> heightmap(width * height, 10.0f);
    
    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use different min/max to avoid division by zero
    // Normalize from [0, 20] to [-1, 1]
    normalizeHeightmap(d_hmap, width, height, 0.0f, 20.0f);
    
    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // All values are 10.0, which in range [0, 20] maps to 0.0 in [-1, 1]
    for (float val : heightmap) {
        EXPECT_NEAR(val, 0.0f, 1e-5f);
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val, 1.0f + 1e-5f);
    }
    
    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, NegativeValues) {
    int width = 16, height = 16;
    std::vector<float> heightmap(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = -static_cast<float>(i % 50);
    }
    
    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);
    
    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    normalizeHeightmap(d_hmap, width, height, min_val, max_val);
    
    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val, 1.0f + 1e-5f);
        EXPECT_TRUE(std::isfinite(val));
    }
    
    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, MixedSigns) {
    int width = 16, height = 16;
    std::vector<float> heightmap(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>((i % 100) - 50);
    }
    
    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);
    
    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    normalizeHeightmap(d_hmap, width, height, min_val, max_val);
    
    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val, 1.0f + 1e-5f);
        EXPECT_TRUE(std::isfinite(val));
    }
    
    cudaFree(d_hmap);
}

TEST(NormalizeHeightmapKernel, RectangularGrid) {
    int width = 32, height = 16;
    std::vector<float> heightmap(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        heightmap[i] = static_cast<float>((i * 13 % 101) - 50);
    }
    
    float min_val, max_val;
    computeMinMax(heightmap, min_val, max_val);
    
    float* d_hmap;
    cudaMalloc(&d_hmap, width * height * sizeof(float));
    cudaMemcpy(d_hmap, heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    normalizeHeightmap(d_hmap, width, height, min_val, max_val);
    
    cudaMemcpy(heightmap.data(), d_hmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float val : heightmap) {
        EXPECT_GE(val, -1.0f - 1e-5f);
        EXPECT_LE(val, 1.0f + 1e-5f);
        EXPECT_TRUE(std::isfinite(val));
    }
    
    cudaFree(d_hmap);
}
