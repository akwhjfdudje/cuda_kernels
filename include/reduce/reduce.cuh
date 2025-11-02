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

#include "reduce_sum.cuh"
