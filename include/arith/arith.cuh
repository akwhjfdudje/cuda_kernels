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

#include "vector_add.cuh"
#include "vector_sub.cuh"
#include "vector_mul.cuh"
#include "vector_div.cuh"
#include "vector_pow.cuh"
#include "vector_sqrt.cuh"
#include "vector_exp.cuh"
#include "vector_log.cuh"
