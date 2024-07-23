#include "ops.h"
#include <immintrin.h>

namespace ops {

void add_cpu(const Tensor& a, const Tensor& b, Tensor& result) {
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    int size = a.shape()[0];

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a_data + i);
        __m256 vb = _mm256_loadu_ps(b_data + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result_data + i, vr);
    }

    for (; i < size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
}

}  // namespace ops