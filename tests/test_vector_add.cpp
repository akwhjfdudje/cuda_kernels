#include <gtest/gtest.h>
#include "vector_add.cuh"
#include <vector>
#include <cmath>

TEST(VectorAddKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Call your CUDA host wrapper (launches the kernel)
    vectorAdd(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-5);
    }
}
