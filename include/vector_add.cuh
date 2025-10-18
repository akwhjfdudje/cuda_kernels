#pragma once

// Launch function to perform vector addition on the GPU
void launchVectorAdd(const float* A, const float* B, float* C, int N);
