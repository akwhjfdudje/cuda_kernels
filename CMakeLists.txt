cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

# === Compiler settings ===
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

find_package(CUDAToolkit REQUIRED)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Debug device flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")

# === Globbing source files ===
file(GLOB src_activate src/activate/*.cu)
file(GLOB src_alg src/alg/*.cu)
file(GLOB src_reduce src/reduce/*.cu)
file(GLOB src_linalg src/linalg/*.cu)
file(GLOB src_conv src/conv/*.cu)
file(GLOB src_arith src/arith/*.cu)
file(GLOB src_fused src/fused/*.cu)
file(GLOB src_image src/image/*.cu)
file(GLOB src_utils src/utils/*.cu)
file(GLOB src_render src/render/*.cu)

set(source
    ${src_activate}
    ${src_alg}
    ${src_reduce}
    ${src_linalg}
    ${src_conv}
    ${src_arith}
    ${src_fused}
    ${src_image}
    ${src_utils}
    ${src_render}
)

file(GLOB test tests/*)

# === CUDA Library ===
add_library(cuda_kernels SHARED ${source})

# Allow linking across multiple .cu files
set_target_properties(cuda_kernels PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Export DLL symbols
target_compile_definitions(cuda_kernels PRIVATE CUDA_KERNELS_EXPORTS)

# Include directories
target_include_directories(cuda_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

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
)

target_link_libraries(cuda_kernel_tests
    PRIVATE cuda_kernels GTest::gtest_main CUDA::cudart
)

target_include_directories(cuda_kernel_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(cuda_kernel_tests)

# === Examples ===
add_executable(examples
    src/examples.cpp
)

target_link_libraries(examples PRIVATE cuda_kernels)
target_include_directories(examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
