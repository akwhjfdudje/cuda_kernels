#pragma once

extern "C" CUDA_KERNELS_API void conv2D(
    const float* input, 
    float* output, 
    const float* kernel,
    int W, int H, int ksize);

