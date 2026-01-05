/**
 * @file conv/generic_conv.cu
 * @brief Generic 2D convolution kernel for single-channel images.
 */

#include <cuda_runtime.h>
#include "conv/conv.cuh"

#define TILE_DIM 16
#define MAX_KERNEL_SIZE 15  // Max allowed convolution kernel size

/**
 * @brief Computes 2D convolution for a single-channel image
 *
 * input: [H, W]
 * kernel: [ksize, ksize]
 * output: [H, W]
 *
 * @param input Pointer to input image
 * @param output Pointer to output image
 * @param kernel Pointer to convolution kernel
 * @param W Image width
 * @param H Image height
 * @param ksize Convolution kernel size (must be odd)
 */
__global__ void conv2DKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ kernel,
    int W, int H,
    int ksize)
{
    __shared__ float tile[TILE_DIM + MAX_KERNEL_SIZE - 1][TILE_DIM + MAX_KERNEL_SIZE - 1];

    int half = ksize / 2;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load input tile into shared memory
    for (int dy = threadIdx.y; dy < TILE_DIM + ksize - 1; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < TILE_DIM + ksize - 1; dx += blockDim.x) {
            int ix = blockIdx.x * TILE_DIM + dx - half;
            int iy = blockIdx.y * TILE_DIM + dy - half;
            ix = min(max(ix, 0), W - 1);  // clamp
            iy = min(max(iy, 0), H - 1);
            tile[dy][dx] = input[iy * W + ix];
        }
    }
    __syncthreads();

    // Only threads inside image bounds compute
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
            float val = tile[threadIdx.y + ky][threadIdx.x + kx];
            float w   = kernel[ky * ksize + kx];
            sum += val * w;
        }
    }

    output[y * W + x] = sum;
}

/**
 * @brief Host launcher for 2D convolution
 */
extern "C" CUDA_KERNELS_API
void conv2D(
    const float* input, 
    float* output, 
    const float* kernel,
    int W, int H, int ksize)
{
    float *dInput = nullptr, *dOutput = nullptr, *dKernel = nullptr;
    size_t imageSize = (size_t)W * H * sizeof(float);
    size_t kernelSize = (size_t)ksize * ksize * sizeof(float);

    cudaMalloc(&dInput, imageSize);
    cudaMalloc(&dOutput, imageSize);
    cudaMalloc(&dKernel, kernelSize);

    cudaMemcpy(dInput, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);

    conv2DKernel<<<blocks, threads>>>(dInput, dOutput, dKernel, W, H, ksize);

    cudaMemcpy(output, dOutput, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFree(dKernel);
}

