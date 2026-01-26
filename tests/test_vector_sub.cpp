#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorSubKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorSub(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] - B[i], 1e-5);
    }
}

TEST(VectorSubKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorSub(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] - B[i], 1e-5);
    }
}

TEST(VectorSubKernel, SingleElement) {
    std::vector<float> A = {10.5f}, B = {3.2f}, C(1);
    vectorSub(A.data(), B.data(), C.data(), 1);
    EXPECT_NEAR(C[0], 7.3f, 1e-5);
}

TEST(VectorSubKernel, ZeroValues) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N, 0.0f), C(N);
    vectorSub(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5);
    }
}

TEST(VectorSubKernel, NegativeValues) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i);
        B[i] = -static_cast<float>(N - i);
    }
    vectorSub(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] - B[i], 1e-5);
    }
}

TEST(VectorSubKernel, IdentityProperty) {
    int N = 256;
    std::vector<float> A(N), B(N, 0.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }
    vectorSub(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

TEST(VectorSubKernel, SelfSubtraction) {
    int N = 100;
    std::vector<float> A(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }
    vectorSub(A.data(), A.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5);
    }
}

TEST(VectorSubKernel, LargeNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i);
        B[i] = 1e6f * static_cast<float>(N - i);
    }
    vectorSub(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] - B[i], 1e-2f);
    }
}
