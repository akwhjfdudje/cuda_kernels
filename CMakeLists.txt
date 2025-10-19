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

# Globbing
file(GLOB src_activate src/activate/*.cu)
file(GLOB src_reduce src/reduce/*.cu)
file(GLOB src_linalg src/linalg/*.cu)
file(GLOB src_conv src/conv/*.cu)
file(GLOB src_arith src/arith/*.cu)
file(GLOB src_image src/image/*.cu)
file(GLOB src_utils src/utils/*.cu)
file(GLOB src_render src/render/*.cu)
file(GLOB source 
    ${src_activate}
    ${src_reduce}
    ${src_linalg}
    ${src_conv}
    ${src_arith}
    ${src_image}
    ${src_utils}
    ${src_render}
)
file(GLOB test tests/*)

# === CUDA Library ===
add_library(cuda_kernels STATIC
    ${source}
)

# Allow linking across multiple .cu files
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# === Unit Tests ===
enable_testing()
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(cuda_kernel_tests
    ${test}
    ${source}
)
target_link_libraries(cuda_kernel_tests
    PRIVATE cuda_kernels GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(cuda_kernel_tests)

# === Executable ===
add_executable(examples
    src/examples.cpp
    src/arith/vector_add.cu
    src/linalg/matrix_mul.cu
    src/reduce/reduce_sum.cu
)

target_link_libraries(examples PRIVATE cuda_kernels)

