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
