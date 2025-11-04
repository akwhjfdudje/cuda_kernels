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
