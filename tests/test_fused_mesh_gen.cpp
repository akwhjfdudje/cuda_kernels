#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "fused/fused.cuh"

/**
 * @brief CPU reference generator for vertices, normals, and texcoords.
 */
static void cpuGenerateMesh(
        const std::vector<float>& hm,
        int W, int H,
        float scaleX,
        float scaleY,
        float heightScale,
        std::vector<float3>& outVerts,
        std::vector<float3>& outNorms,
        std::vector<float2>& outUV)
{
    int N = W * H;
    outVerts.resize(N);
    outNorms.resize(N);
    outUV.resize(N);

    auto getH = [&](int x, int y) {
        return hm[y * W + x];
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {

            int idx = y * W + x;
            float h = getH(x, y);

            // vertex
            float vx = x * scaleX;
            float vy = h * heightScale;
            float vz = y * scaleY;
            outVerts[idx] = make_float3(vx, vy, vz);

            // texcoord
            float u = (W > 1) ? float(x) / float(W - 1) : 0.0f;
            float v = (H > 1) ? float(y) / float(H - 1) : 0.0f;
            outUV[idx] = make_float2(u, v);

            // neighbors (clamped)
            int xm = (x > 0)     ? x - 1 : x;
            int xp = (x < W - 1) ? x + 1 : x;
            int ym = (y > 0)     ? y - 1 : y;
            int yp = (y < H - 1) ? y + 1 : y;

            float hL = getH(xm, y);
            float hR = getH(xp, y);
            float hD = getH(x, ym);
            float hU = getH(x, yp);

            float dx = (hR - hL) * 0.5f * heightScale / ((scaleX != 0.0f) ? scaleX : 1.0f);
            float dz = (hU - hD) * 0.5f * heightScale / ((scaleY != 0.0f) ? scaleY : 1.0f);

            float3 n;
            n.x = -dx;
            n.y =  1.0f;
            n.z = -dz;

            float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
            if (len > 1e-9f) {
                n.x /= len; n.y /= len; n.z /= len;
            }
            outNorms[idx] = n;
        }
    }
}

/**
 * @brief Small grid test.
 */
TEST(FusedMeshKernel, SmallGrid) {

    int W = 8;
    int H = 6;
    int N = W * H;

    float scaleX = 0.1f;
    float scaleY = 0.2f;
    float heightScale = 5.0f;

    // host heightmap
    std::vector<float> hm(N);
    for (int i = 0; i < N; ++i)
        hm[i] = 0.05f * ((i * 7) % 23);  // pseudo-random small values

    // device memory
    float* d_hm;
    float3 *d_v, *d_n;
    float2 *d_uv;

    cudaMalloc(&d_hm, N * sizeof(float));
    cudaMalloc(&d_v,  N * sizeof(float3));
    cudaMalloc(&d_n,  N * sizeof(float3));
    cudaMalloc(&d_uv, N * sizeof(float2));

    cudaMemcpy(d_hm, hm.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    generateMeshFromHeightmap(
        d_hm, d_v, d_n, d_uv,
        W, H,
        scaleX, scaleY, heightScale
    );

    // Copy back
    std::vector<float3> v(N), n(N);
    std::vector<float2> uv(N);
    cudaMemcpy(v.data(),  d_v,  N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(n.data(),  d_n,  N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(uv.data(), d_uv, N*sizeof(float2), cudaMemcpyDeviceToHost);

    // CPU reference
    std::vector<float3> refV, refN;
    std::vector<float2> refUV;
    cpuGenerateMesh(hm, W, H, scaleX, scaleY, heightScale, refV, refN, refUV);

    // Compare
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(v[i].x, refV[i].x, 1e-5f);
        EXPECT_NEAR(v[i].y, refV[i].y, 1e-5f);
        EXPECT_NEAR(v[i].z, refV[i].z, 1e-5f);

        EXPECT_NEAR(uv[i].x, refUV[i].x, 1e-6f);
        EXPECT_NEAR(uv[i].y, refUV[i].y, 1e-6f);

        EXPECT_NEAR(n[i].x, refN[i].x, 1e-5f);
        EXPECT_NEAR(n[i].y, refN[i].y, 1e-5f);
        EXPECT_NEAR(n[i].z, refN[i].z, 1e-5f);
    }

    cudaFree(d_hm);
    cudaFree(d_v);
    cudaFree(d_n);
    cudaFree(d_uv);
}

/**
 * @brief Moderate grid test.
 */
TEST(FusedMeshKernel, ModerateGrid) {

    int W = 64;
    int H = 48;
    int N = W * H;

    float scaleX = 0.02f;
    float scaleY = 0.03f;
    float heightScale = 10.0f;

    std::vector<float> hm(N);
    for (int i = 0; i < N; ++i)
        hm[i] = 0.01f * ((i % 29) - 14);

    float *d_hm;
    float3 *d_v, *d_n;
    float2 *d_uv;

    cudaMalloc(&d_hm, N * sizeof(float));
    cudaMalloc(&d_v,  N * sizeof(float3));
    cudaMalloc(&d_n,  N * sizeof(float3));
    cudaMalloc(&d_uv, N * sizeof(float2));
    cudaMemcpy(d_hm, hm.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    generateMeshFromHeightmap(
        d_hm, d_v, d_n, d_uv,
        W, H,
        scaleX, scaleY, heightScale
    );

    std::vector<float3> v(N), n(N);
    std::vector<float2> uv(N);
    cudaMemcpy(v.data(),  d_v,  N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(n.data(),  d_n,  N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(uv.data(), d_uv, N*sizeof(float2), cudaMemcpyDeviceToHost);

    std::vector<float3> refV, refN;
    std::vector<float2> refUV;
    cpuGenerateMesh(hm, W, H, scaleX, scaleY, heightScale, refV, refN, refUV);

    for (int i = 0; i < N; ++i) {
        float tolV = 1e-4f;
        float tolN = 1e-4f;

        EXPECT_NEAR(v[i].x, refV[i].x, tolV);
        EXPECT_NEAR(v[i].y, refV[i].y, tolV);
        EXPECT_NEAR(v[i].z, refV[i].z, tolV);

        EXPECT_NEAR(uv[i].x, refUV[i].x, 1e-6f);
        EXPECT_NEAR(uv[i].y, refUV[i].y, 1e-6f);

        EXPECT_NEAR(n[i].x, refN[i].x, tolN);
        EXPECT_NEAR(n[i].y, refN[i].y, tolN);
        EXPECT_NEAR(n[i].z, refN[i].z, tolN);
    }

    cudaFree(d_hm);
    cudaFree(d_v);
    cudaFree(d_n);
    cudaFree(d_uv);
}

