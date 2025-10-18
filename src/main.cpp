#include "vector_add.cuh"
#include <iostream>
#include <vector>

int main() {
    int N = 1 << 20; // 1M elements
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    launchVectorAdd(A.data(), B.data(), C.data(), N);

    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "C[N-1] = " << C[N-1] << std::endl;
    return 0;
}
