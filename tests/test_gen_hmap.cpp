#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "render/render.cuh"

inline float hash_cpu(int x, int y, int seed = 1337) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

inline float fade_cpu(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
inline float lerp_cpu(float a, float b, float t) { return a + (b - a) * t; }

float perlin_cpu(float x, float y, int seed) {
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float sx = fade_cpu(x - x0);
    float sy = fade_cpu(y - y0);
    float n00 = hash_cpu(x0, y0, seed);
    float n10 = hash_cpu(x1, y0, seed);
    float n01 = hash_cpu(x0, y1, seed);
    float n11 = hash_cpu(x1, y1, seed);
    float ix0 = lerp_cpu(n00, n10, sx);
    float ix1 = lerp_cpu(n01, n11, sx);
    return lerp_cpu(ix0, ix1, sy);
}

float voronoi_cpu(float x, float y, int cell_count, int seed) {
    int xi = static_cast<int>(floorf(x));
    int yi = static_cast<int>(floorf(y));
    float minDist = 1e10f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int cx = xi + i;
            int cy = yi + j;
            float rx = hash_cpu(cx, cy, seed) * 0.5f + 0.5f;
            float ry = hash_cpu(cx, cy, seed + 42) * 0.5f + 0.5f;
            float dx = (cx + rx) - x;
            float dy = (cy + ry) - y;
            minDist = std::min(minDist, std::sqrt(dx * dx + dy * dy));
        }
    }
    return minDist;
}

void cpu_generateHeightmap(std::vector<float>& out, int width, int height,
                           float scale, int seed, float mix_ratio) {
    for (int idx = 0; idx < width * height; ++idx) {
        int x = idx % width;
        int y = idx / width;
        float fx = x / scale;
        float fy = y / scale;

        float amp = 1.0f, freq = 1.0f, noise_sum = 0.0f;
        for (int o = 0; o < 4; ++o) {
            noise_sum += perlin_cpu(fx * freq, fy * freq, seed + o * 17) * amp;
            amp *= 0.5f;
            freq *= 2.0f;
        }
        float v = voronoi_cpu(fx * 0.5f, fy * 0.5f, 8, seed + 999);
        float height_val = (1.0f - mix_ratio) * noise_sum + mix_ratio * (1.0f - v * 2.0f);
        out[idx] = std::max(-1.0f, std::min(1.0f, height_val));
    }
}

TEST(GenerateHeightmap, SmallGrid) {
    int width = 8, height = 8;
    float scale = 4.0f;
    int seed = 42;
    float mix_ratio = 0.3f;
    float tol = 1e-4f;

    std::vector<float> cpuOut(width * height);
    std::vector<float> gpuOut(width * height);

    cpu_generateHeightmap(cpuOut, width, height, scale, seed, mix_ratio);
    generateHeightmap(gpuOut.data(), width, height, scale, seed, mix_ratio);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_NEAR(gpuOut[i], cpuOut[i], tol) << "Mismatch at index " << i;
    }
}

TEST(GenerateHeightmap, ModerateGrid) {
    int width = 32, height = 32;
    float scale = 8.0f;
    int seed = 123;
    float mix_ratio = 0.6f;
    float tol = 1e-4f;

    std::vector<float> cpuOut(width * height);
    std::vector<float> gpuOut(width * height);

    cpu_generateHeightmap(cpuOut, width, height, scale, seed, mix_ratio);
    generateHeightmap(gpuOut.data(), width, height, scale, seed, mix_ratio);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_NEAR(gpuOut[i], cpuOut[i], tol) << "Mismatch at index " << i;
    }
}

