#include <gtest/gtest.h>
#include "conv/conv.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @brief CPU reference implementation of 2D convolution.
 *        Assumes clamped boundary conditions.
 */
static void cpuConv2D(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int W, int H, int ksize)
{
    output.resize(W * H);
    int half = ksize / 2;

    auto idx = [W](int x, int y) { return y * W + x; };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;

            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = x + kx - half;
                    int iy = y + ky - half;

                    // Clamp to edges
                    ix = std::min(std::max(ix, 0), W - 1);
                    iy = std::min(std::max(iy, 0), H - 1);

                    sum += input[idx(ix, iy)] * kernel[ky * ksize + kx];
                }
            }
            output[idx(x, y)] = sum;
        }
    }
}

/**
 * @brief Small 5x5 image with 3x3 kernel.
 */
TEST(Conv2DKernel, SmallImage) {
    int W = 5;
    int H = 5;
    int ksize = 3;

    std::vector<float> input(W * H);
    std::vector<float> kernel(ksize * ksize);
    std::vector<float> output(W * H);
    std::vector<float> ref(W * H);

    // Initialize input and kernel with simple values
    for (int i = 0; i < W * H; ++i)
        input[i] = float(i % 7);

    for (int i = 0; i < ksize * ksize; ++i)
        kernel[i] = 0.1f * (i + 1);

    // Run CUDA kernel
    conv2D(input.data(), output.data(), kernel.data(), W, H, ksize);

    // Run CPU reference
    cpuConv2D(input, kernel, ref, W, H, ksize);

    // Compare results
    for (int i = 0; i < W * H; ++i) {
        EXPECT_NEAR(output[i], ref[i], 1e-5f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Moderate size image and kernel, more realistic.
 */
TEST(Conv2DKernel, ModerateImage) {
    int W = 32;
    int H = 24;
    int ksize = 5;

    std::vector<float> input(W * H);
    std::vector<float> kernel(ksize * ksize);
    std::vector<float> output(W * H);
    std::vector<float> ref(W * H);

    // Fill input with small random-ish values
    for (int i = 0; i < W * H; ++i)
        input[i] = 0.1f * (i % 13);

    // Example kernel: simple averaging
    for (int i = 0; i < ksize * ksize; ++i)
        kernel[i] = 1.0f / (ksize * ksize);

    conv2D(input.data(), output.data(), kernel.data(), W, H, ksize);
    cpuConv2D(input, kernel, ref, W, H, ksize);

    for (int i = 0; i < W * H; ++i) {
        float tol = 1e-4f * std::max(1.0f, std::fabs(ref[i]));
        EXPECT_NEAR(output[i], ref[i], tol) << "Mismatch at index " << i;
    }
}

