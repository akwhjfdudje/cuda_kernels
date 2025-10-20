#include <gtest/gtest.h>
#include <vector>
#include "activate/activate.cuh"

TEST(VectorReLUKernel, BasicPositiveAndNegative) {
    int N = 8;
    std::vector<float> A = {-2.0f, -1.0f, 0.0f, 1.0f, 2.5f, -0.3f, 4.0f, -5.0f};
    std::vector<float> C(N, 0.0f);

    vectorReLU(A.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::max(0.0f, A[i]);
        EXPECT_NEAR(C[i], expected, 1e-6f) << "Mismatch at index " << i;
    }
}

TEST(VectorReLUKernel, AllNegatives) {
    int N = 4;
    std::vector<float> A = {-10.0f, -0.5f, -3.3f, -7.1f};
    std::vector<float> C(N, 0.0f);

    vectorReLU(A.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(C[i], 0.0f);
    }
}

TEST(VectorReLUKernel, LargeArrayFiniteValues) {
    int N = 1 << 16;
    std::vector<float> A(N), C(N);

    for (int i = 0; i < N; ++i)
        A[i] = (i % 200 - 100) * 0.05f; // mix of neg/pos

    vectorReLU(A.data(), C.data(), N);

    for (float v : C) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

