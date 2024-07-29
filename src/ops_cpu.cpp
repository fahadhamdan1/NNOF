#include "ops.h"
#include <immintrin.h>

namespace ops {

void add_cpu_baseline(const Tensor& a, const Tensor& b, Tensor& result) {
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    int size = a.shape()[0] * a.shape()[1];

    for (int i = 0; i < size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
}


void add_cpu(const Tensor& a, const Tensor& b, Tensor& result) {
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    int size = a.shape()[0] * a.shape()[1];

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

void matmul_cpu_baseline(const Tensor& a, const Tensor& b, Tensor& result) {
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }
}

void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& result) {
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];

    // Initialize result to zero
    std::fill(result_data, result_data + m * n, 0.0f);

    const int block_size = 64;

    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int l = 0; l < k; l += block_size) {
                int max_i = std::min(i + block_size, m);
                int max_j = std::min(j + block_size, n);
                int max_l = std::min(l + block_size, k);

                for (int ii = i; ii < max_i; ++ii) {
                    for (int jj = j; jj < max_j; jj += 8) {
                        __m256 sum = _mm256_loadu_ps(&result_data[ii * n + jj]);
                        for (int ll = l; ll < max_l; ++ll) {
                            __m256 a_val = _mm256_set1_ps(a_data[ii * k + ll]);
                            __m256 b_val = _mm256_loadu_ps(&b_data[ll * n + jj]);
                            sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_val));
                        }
                        _mm256_storeu_ps(&result_data[ii * n + jj], sum);
                    }
                    
                    for (int jj = (max_j / 8) * 8; jj < max_j; ++jj) {
                        float sum = result_data[ii * n + jj];
                        for (int ll = l; ll < max_l; ++ll) {
                            sum += a_data[ii * k + ll] * b_data[ll * n + jj];
                        }
                        result_data[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

}