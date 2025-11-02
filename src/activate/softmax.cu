/**
 * @file activate/softmax.cu
 * @brief CUDA kernel for computing row-wise softmax.
 */

#include <cuda_runtime.h>
#include <math.h>
#include "activate/activate.cuh"

#define THREADS_PER_BLOCK 256

/**
 * @brief CUDA kernel computing softmax for each row of a 2D tensor.
 *
 * @param input Pointer to input matrix (size: batch_size x features)
 * @param output Pointer to output matrix (same size as input)
 * @param batch_size Number of rows
 * @param features Number of columns per row
 */
__global__ void softmaxKernel(const float* input, float* output, int batch_size, int features) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * features;

    if (row >= batch_size) return;

    // --- Step 1: Compute max value in the row for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < features; i += blockDim.x)
        max_val = fmaxf(max_val, input[offset + i]);

    // Parallel reduction for max
    shared[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    // --- Step 2: Compute exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        float val = expf(input[offset + i] - max_val);
        output[offset + i] = val;
        sum += val;
    }

    // Parallel reduction for sum
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();

    // --- Step 3: Normalize
    for (int i = tid; i < features; i += blockDim.x) {
        output[offset + i] /= sum;
    }
}

/**
 * @brief Host launcher for softmax kernel.
 *
 * @param input Pointer to host input array (size: batch_size x features)
 * @param output Pointer to host output array (same size)
 * @param batch_size Number of rows
 * @param features Number of columns per row
 */
extern "C" CUDA_KERNELS_API void softmax(const float* input, float* output, int batch_size, int features) {
    float *d_in, *d_out;
    size_t size = batch_size * features * sizeof(float);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 blocks(batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    size_t shared_size = THREADS_PER_BLOCK * sizeof(float);

    softmaxKernel<<<blocks, threads, shared_size>>>(d_in, d_out, batch_size, features);

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
