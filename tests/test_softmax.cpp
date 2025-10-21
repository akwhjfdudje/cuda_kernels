#include <gtest/gtest.h>
#include "activate/activate.cuh"
#include <vector>
#include <math.h>
#include <float.h>

/**
 * @brief Reference CPU implementation of row-wise softmax.
 */
static void softmax_CPU(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; ++j)
            max_val = fmaxf(max_val, input[i * cols + j]);

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float e = expf(input[i * cols + j] - max_val);
            output[i * cols + j] = e;
            sum += e;
        }

        for (int j = 0; j < cols; ++j)
            output[i * cols + j] /= sum;
    }
}

/**
 * @brief Basic stability test for the softmax2D kernel.
 */
TEST(SoftmaxKernel, BasicOutput) {
    const int rows = 4;
    const int cols = 8;
    std::vector<float> input(rows * cols), output(rows * cols);

    for (int i = 0; i < rows * cols; ++i)
        input[i] = static_cast<float>(i - cols / 2);

    softmax(input.data(), output.data(), rows, cols);

    for (float val : output)
        EXPECT_TRUE(std::isfinite(val));
}

/**
 * @brief Correctness test comparing GPU result to CPU reference.
 */
TEST(SoftmaxKernel, BasicCorrectness) {
    const int rows = 3;
    const int cols = 5;
    std::vector<float> input(rows * cols), output(rows * cols), ref(rows * cols);

    // Initialize input with predictable pattern
    for (int i = 0; i < rows * cols; ++i)
        input[i] = static_cast<float>((i % cols) - 2);

    // Run CPU reference
    softmax_CPU(input.data(), ref.data(), rows, cols);

    // Run GPU version
    softmax(input.data(), output.data(), rows, cols);

    // Check that each row sums to ~1 and values are nonnegative
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            sum += output[idx];
            EXPECT_GE(output[idx], 0.0f);
            EXPECT_NEAR(output[idx], ref[idx], 1e-4f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-3f);
    }
}
