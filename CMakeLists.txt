cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

# Set runtime library at top
if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# === Build Settings ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)


# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Enable warnings
add_compile_options()

# === CUDA Library ===
add_library(cuda_kernels STATIC
    src/vector_add.cu
    src/matrix_mul.cu
    src/reduce.cu
)

# Allow linking across multiple .cu files
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# === Executable ===
add_executable(examples
    src/main.cpp
    src/vector_add.cu
    src/matrix_mul.cu
    src/reduce.cu
)

target_link_libraries(examples PRIVATE cuda_kernels)

