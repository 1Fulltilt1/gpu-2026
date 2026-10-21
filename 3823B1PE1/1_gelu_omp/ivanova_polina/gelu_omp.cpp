#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        
        float exp_2y = std::exp(2.0f * inner);
        float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
        
        output[i] = 0.5f * x * (1.0f + fast_tanh);
    }

    return output;
}