#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <vector>
#include <math.h>
#include "activate/activate.cuh"

static float cpu_gelu(float x) {
    const float k = sqrtf(2.0f / M_PI);
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

TEST(VectorGELUKernel, BasicValues) {
    int N = 5;
    std::vector<float> A = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> C(N, 0.0f);

    vectorGELU(A.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = cpu_gelu(A[i]);
        EXPECT_NEAR(C[i], expected, 1e-4f) << "Mismatch at index " << i;
    }
}

TEST(VectorGELUKernel, SmoothnessCheck) {
    int N = 1000;
    std::vector<float> A(N), C(N);

    for (int i = 0; i < N; ++i)
        A[i] = -3.0f + 6.0f * i / (float)(N - 1); // [-3, 3]

    vectorGELU(A.data(), C.data(), N);

    for (float v : C) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(VectorGELUKernel, ZeroInput) {
    int N = 32;
    std::vector<float> A(N, 0.0f), C(N);
    vectorGELU(A.data(), C.data(), N);
    for (float v : C) {
        EXPECT_NEAR(v, 0.0f, 1e-7f);
    }
}

