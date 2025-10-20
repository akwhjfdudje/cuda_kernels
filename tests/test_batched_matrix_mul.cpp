#include <gtest/gtest.h>
#include "linalg/linalg.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @brief CPU reference implementation for batched matrix multiplication.
 */
static void cpuBatchedMatMul(const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int N, int batch)
{
    for (int b = 0; b < batch; ++b) {
        const float* batchA = &A[b * N * N];
        const float* batchB = &B[b * N * N];
        float* batchC = &C[b * N * N];

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                    sum += batchA[i * N + k] * batchB[k * N + j];
                batchC[i * N + j] = sum;
            }
        }
    }
}

/**
 * @brief Test small batch matrix multiplication for correctness.
 */
TEST(BatchedMatrixMulKernel, SmallBatch) {
    int N = 8;
    int batch = 4;
    std::vector<float> A(batch * N * N);
    std::vector<float> B(batch * N * N);
    std::vector<float> C(batch * N * N);
    std::vector<float> C_ref(batch * N * N);

    // Initialize matrices with simple values
    for (int i = 0; i < batch * N * N; ++i) {
        A[i] = static_cast<float>(i % 13);
        B[i] = static_cast<float>((i * 3) % 7);
    }

    batchedMatrixMul(A.data(), B.data(), C.data(), N, batch);
    cpuBatchedMatMul(A, B, C_ref, N, batch);

    for (int i = 0; i < batch * N * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test moderately large batch for stability and performance.
 */
TEST(BatchedMatrixMulKernel, ModerateBatch) {
    int N = 32;
    int batch = 16;
    std::vector<float> A(batch * N * N);
    std::vector<float> B(batch * N * N);
    std::vector<float> C(batch * N * N);
    std::vector<float> C_ref(batch * N * N);

    // Fill with small random values to avoid overflow
    for (int i = 0; i < batch * N * N; ++i) {
        A[i] = 0.1f * (i % 10);
        B[i] = 0.1f * ((i + 3) % 10);
    }

    batchedMatrixMul(A.data(), B.data(), C.data(), N, batch);
    cpuBatchedMatMul(A, B, C_ref, N, batch);

    for (int i = 0; i < batch * N * N; ++i) {
        float expected = C_ref[i];
        float tol = 1e-3f * std::max(1.0f, fabs(expected));
        EXPECT_NEAR(C[i], expected, tol) << "Mismatch at index " << i;
    }
}
