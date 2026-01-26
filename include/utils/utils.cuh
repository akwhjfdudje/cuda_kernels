#pragma once

/**
 * @file utils/utils.cuh
 * @brief Shared utility functions for CUDA kernels.
 */

/**
 * @brief Device-side Linear Congruential Generator (LCG) for random number generation.
 * 
 * This function implements a fast, deterministic, device-safe RNG using the LCG algorithm.
 * It works entirely on the GPU without any global state, making it suitable for parallel
 * random number generation across multiple threads.
 * 
 * Algorithm:
 *   state = (a*state + c) mod m
 *   - 'state' is the current RNG state (seed), passed by reference.
 *   - 'a' is the multiplier constant (1664525u), chosen for good statistical properties.
 *   - 'c' is the increment constant (1013904223u), also chosen for LCG quality.
 *   - 'm' is implicitly 2^32 due to 32-bit unsigned integer overflow.
 * 
 * The resulting 'state' is then converted to a floating-point number in [0,1):
 *   - Mask the lower 24 bits with 0x00FFFFFF to avoid using all 32 bits.
 *   - Divide by 2^24 (0x01000000) to normalize to the range [0,1).
 * 
 * @param state Reference to the RNG state (seed), which is updated by this function.
 * @return A random float value in the range [0, 1).
 * 
 * @note This function is thread-safe as each thread should maintain its own state variable.
 * @note The same seed will always produce the same sequence of values, making this
 *       suitable for reproducible random number generation.
 */
__device__ inline float lcg_random(unsigned int& state) {
    // Update RNG state using LCG formula
    state = state * 1664525u + 1013904223u;
    
    // Convert to [0,1) using lower 24 bits
    return (state & 0x00FFFFFF) / float(0x01000000);
}
