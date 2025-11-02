#ifdef _WIN32
  #ifdef CUDA_KERNELS_EXPORTS
    #define CUDA_KERNELS_API __declspec(dllexport)
  #else
    #define CUDA_KERNELS_API __declspec(dllimport)
  #endif
#else
  #define CUDA_KERNELS_API
#endif
#pragma once

#include "vector_gelu.cuh"
#include "vector_relu.cuh"
#include "softmax.cuh"
