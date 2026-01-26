/**
 * @file test_normals.cu
 * @brief Unit tests for computeNormals CUDA kernel using GoogleTest.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "render/render.cuh"

/**
 * @brief Test that computeNormals produces approximately correct normals on a small 3x3 heightmap.
 *
 * This test creates a simple heightmap where the center is higher than its neighbors.
 * The expected normal at the center should point roughly upward (0,1,0).
 */
TEST(ComputeNormalsKernel, SmallHeightmap) {
    const int width = 3;
    const int height = 3;
    const int total = width * height;

    std::vector<float> h_heightmap = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    std::vector<float3> h_normals(total);

    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;

    cudaMalloc(&d_heightmap, total * sizeof(float));
    cudaMalloc(&d_normals, total * sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    computeNormals(d_heightmap, d_normals, width, height);

    cudaMemcpy(h_normals.data(), d_normals, total * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(d_heightmap);
    cudaFree(d_normals);

    // Check the center normal (index 4)
    float3 center = h_normals[4];
    float len = std::sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
    EXPECT_NEAR(len, 1.0f, 1e-3f);

    // Should be mostly upward
    EXPECT_GT(center.y, 0.7f);

    // Edge normals should not be NaN
    for (int i = 0; i < total; ++i) {
        EXPECT_TRUE(std::isfinite(h_normals[i].x));
        EXPECT_TRUE(std::isfinite(h_normals[i].y));
        EXPECT_TRUE(std::isfinite(h_normals[i].z));
    }
}

/**
 * @brief Test that computeNormals handles a flat plane correctly.
 *
 * All normals should point straight up (0,1,0) since there is no height variation.
 */
TEST(ComputeNormalsKernel, FlatPlane) {
    const int width = 4;
    const int height = 4;
    const int total = width * height;

    std::vector<float> h_heightmap(total, 1.0f);
    std::vector<float3> h_normals(total);

    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;

    cudaMalloc(&d_heightmap, total * sizeof(float));
    cudaMalloc(&d_normals, total * sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    computeNormals(d_heightmap, d_normals, width, height);

    cudaMemcpy(h_normals.data(), d_normals, total * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(d_heightmap);
    cudaFree(d_normals);

    for (int i = 0; i < total; ++i) {
        float3 n = h_normals[i];
        EXPECT_NEAR(n.x, 0.0f, 1e-4f);
        EXPECT_NEAR(n.y, 1.0f / std::sqrt(1.0f + 0.0f + 0.0f), 1e-4f);
        EXPECT_NEAR(n.z, 0.0f, 1e-4f);
    }
}

TEST(ComputeNormalsKernel, SinglePixel) {
    const int width = 1, height = 1;
    std::vector<float> h_heightmap = {1.0f};
    std::vector<float3> h_normals(1);
    
    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;
    
    cudaMalloc(&d_heightmap, sizeof(float));
    cudaMalloc(&d_normals, sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), sizeof(float), cudaMemcpyHostToDevice);
    
    computeNormals(d_heightmap, d_normals, width, height);
    
    cudaMemcpy(h_normals.data(), d_normals, sizeof(float3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_heightmap);
    cudaFree(d_normals);
    
    float3 n = h_normals[0];
    float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    EXPECT_NEAR(len, 1.0f, 1e-3f);
    EXPECT_TRUE(std::isfinite(n.x));
    EXPECT_TRUE(std::isfinite(n.y));
    EXPECT_TRUE(std::isfinite(n.z));
}

TEST(ComputeNormalsKernel, Slope) {
    const int width = 5, height = 5;
    std::vector<float> h_heightmap(width * height);
    
    // Create a slope going from left to right
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_heightmap[y * width + x] = static_cast<float>(x);
        }
    }
    
    std::vector<float3> h_normals(width * height);
    
    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;
    
    cudaMalloc(&d_heightmap, width * height * sizeof(float));
    cudaMalloc(&d_normals, width * height * sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    computeNormals(d_heightmap, d_normals, width, height);
    
    cudaMemcpy(h_normals.data(), d_normals, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_heightmap);
    cudaFree(d_normals);
    
    // Check that normals are normalized
    for (int i = 0; i < width * height; ++i) {
        float3 n = h_normals[i];
        float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        EXPECT_NEAR(len, 1.0f, 1e-3f);
        EXPECT_TRUE(std::isfinite(n.x));
        EXPECT_TRUE(std::isfinite(n.y));
        EXPECT_TRUE(std::isfinite(n.z));
    }
}

TEST(ComputeNormalsKernel, Valley) {
    const int width = 5, height = 5;
    std::vector<float> h_heightmap(width * height);
    
    // Create a valley (low in center, high on edges)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dx = x - width / 2;
            int dy = y - height / 2;
            h_heightmap[y * width + x] = static_cast<float>(dx * dx + dy * dy);
        }
    }
    
    std::vector<float3> h_normals(width * height);
    
    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;
    
    cudaMalloc(&d_heightmap, width * height * sizeof(float));
    cudaMalloc(&d_normals, width * height * sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    computeNormals(d_heightmap, d_normals, width, height);
    
    cudaMemcpy(h_normals.data(), d_normals, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_heightmap);
    cudaFree(d_normals);
    
    // Center should have upward normal
    int center_idx = (height / 2) * width + (width / 2);
    float3 center = h_normals[center_idx];
    EXPECT_GT(center.y, 0.0f);
    
    // All normals should be normalized
    for (int i = 0; i < width * height; ++i) {
        float3 n = h_normals[i];
        float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        EXPECT_NEAR(len, 1.0f, 1e-3f);
    }
}

TEST(ComputeNormalsKernel, LargeHeightmap) {
    const int width = 64, height = 64;
    std::vector<float> h_heightmap(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        h_heightmap[i] = static_cast<float>(i % 100);
    }
    
    std::vector<float3> h_normals(width * height);
    
    float *d_heightmap = nullptr;
    float3 *d_normals = nullptr;
    
    cudaMalloc(&d_heightmap, width * height * sizeof(float));
    cudaMalloc(&d_normals, width * height * sizeof(float3));
    cudaMemcpy(d_heightmap, h_heightmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    computeNormals(d_heightmap, d_normals, width, height);
    
    cudaMemcpy(h_normals.data(), d_normals, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_heightmap);
    cudaFree(d_normals);
    
    // Check a sample of normals
    for (int i = 0; i < 100; ++i) {
        int idx = i * (width * height / 100);
        float3 n = h_normals[idx];
        float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        EXPECT_NEAR(len, 1.0f, 1e-3f);
        EXPECT_TRUE(std::isfinite(n.x));
        EXPECT_TRUE(std::isfinite(n.y));
        EXPECT_TRUE(std::isfinite(n.z));
    }
}
