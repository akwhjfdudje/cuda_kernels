#pragma once

extern "C" CUDA_KERNELS_API void softmax(const float* input, float* output, int batch_size, int features);
