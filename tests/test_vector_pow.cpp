#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorPowKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorPow(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = powf(A[i], B[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));  // relative tolerance
        EXPECT_NEAR(C[i], expected, tol);
    }
}

TEST(VectorPowKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = 0.5f + static_cast<float>(i % 100); 
        B[i] = 0.0f + static_cast<float>(i % 50);   
    }

    vectorPow(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = powf(A[i], B[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));  // relative tolerance
        EXPECT_NEAR(C[i], expected, tol);
    }
}

TEST(VectorPowKernel, SingleElement) {
    std::vector<float> A = {2.0f}, B = {3.0f}, C(1);
    vectorPow(A.data(), B.data(), C.data(), 1);
    EXPECT_NEAR(C[0], 8.0f, 1e-5);
}

TEST(VectorPowKernel, PowerOfZero) {
    int N = 100;
    std::vector<float> A(N), B(N, 0.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 1.0f, 1e-5);
    }
}

TEST(VectorPowKernel, PowerOfOne) {
    int N = 100;
    std::vector<float> A(N), B(N, 1.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

TEST(VectorPowKernel, BaseOfOne) {
    int N = 100;
    std::vector<float> A(N, 1.0f), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        B[i] = static_cast<float>(i);
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 1.0f, 1e-5);
    }
}

TEST(VectorPowKernel, SquareRoot) {
    int N = 100;
    std::vector<float> A(N), B(N, 0.5f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>((i + 1) * (i + 1));
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        EXPECT_NEAR(C[i], expected, 1e-4f);
    }
}

TEST(VectorPowKernel, NegativeExponents) {
    int N = 50;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = -1.0f;
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = powf(A[i], B[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(C[i], expected, tol);
    }
}

TEST(VectorPowKernel, SmallBases) {
    int N = 100;
    std::vector<float> A(N), B(N, 2.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 0.1f + 0.01f * static_cast<float>(i);
    }
    vectorPow(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = powf(A[i], B[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(C[i], expected, tol);
    }
}
