#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    const std::size_t n = input.size();
    std::vector<float> result(n);

    if (n == 0) {
        return result;
    }

    const float* __restrict__ in = input.data();
    float* __restrict__ out = result.data();

    // kScale = -2.0f * sqrt(2/pi)
    constexpr float kScale = -1.5957691216057308f;
    constexpr float kCoef = 0.044715f;

    #pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];

        const float poly = std::fmaf(kCoef, x * x, 1.0f);
        const float arg = kScale * x * poly;
        
        out[i] = x / (1.0f + std::exp(arg));
    }

    return result;
}