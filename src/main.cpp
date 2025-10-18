#include <iostream>
#include <vector>
#include "vector_add.cuh"
#include "matrix_mul.cuh"
#include "reduce.cuh"

int main() {
    // Vector Add
    const int N = 1 << 20;
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);
    vectorAdd(A.data(), B.data(), C.data(), N);
    std::cout << "VectorAdd: C[0] = " << C[0] << ", C[N-1] = " << C[N-1] << "\n";

    // Matrix Multiply
    const int M = 256;
    std::vector<float> MA(M*M, 1.0f), MB(M*M, 2.0f), MC(M*M);
    matrixMul(MA.data(), MB.data(), MC.data(), M);
    std::cout << "MatrixMul: C[0] = " << MC[0] << ", C[M*M-1] = " << MC.back() << "\n";

    // Reduction
    float sum = reduceSum(A.data(), N);
    std::cout << "ReduceSum: total = " << sum << "\n";

    return 0;
}
