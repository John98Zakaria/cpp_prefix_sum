#include <immintrin.h>
#include <iostream>
#include <vector>

int main() {
    constexpr auto shift = 0;
    constexpr auto avx_register_size = 4;
    constexpr auto size = 4;
    std::vector<u_int32_t> data{0, 1, 2, 3, 4, 5, 6, 7};
    __m256i a = _mm256_loadu_si256((__m256i *) data.data());
    auto right_shift = _mm256_slli_si256(a, shift * size);
    auto left_shift = _mm256_srli_si256(a, (avx_register_size - shift) * size);
    auto merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
    auto correct_shift = _mm256_or_si256(right_shift, merge_mask);

    // write a into data
    _mm256_storeu_si256((__m256i *) data.data(), correct_shift);
    for (int i = 0; i < 8; i++) {
        std::cout << data[i] << " ";
    }

    return 0;
}
