//
// Created by jn98zk on 21.04.24.
//
#ifndef AVX2_PREFIX_SUM_AVX2_PREFIX_HPP
#define AVX2_PREFIX_SUM_AVX2_PREFIX_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <execution>
#include <immintrin.h>

template<typename T>
std::vector<T> prefix_sum(std::vector<T> current_iteration_input) {
    std::vector<T> next_iteration_input(current_iteration_input);

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = current_iteration_input.size();
    auto const number_of_iterations = static_cast<uint8_t>(std::ceil(std::log2(vector_size)));
    for (uint8_t iteration_number = 0; iteration_number < number_of_iterations; ++iteration_number) {

        // Copy Routine
        const uint64_t boundary_size = 1 << iteration_number;
        const uint64_t shift_size = boundary_size;

        for (uint64_t i = 0; i < boundary_size; ++i) {

            next_iteration_input[i] = current_iteration_input[i];
        }

        // Prefix Routine
        for (uint64_t i = 0; i < vector_size - boundary_size; ++i) {
            next_iteration_input[i + shift_size] = current_iteration_input[i] + current_iteration_input[i + shift_size];
        }


        std::swap(current_iteration_input, next_iteration_input);
    }


    return current_iteration_input;
}

/**
 * Inplace Prefix Sum using SSE2
 * @param in_out_vector
 */
void inplace_prefix_sum(std::vector<uint8_t> &in_out_vector) {

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = in_out_vector.size();
    auto const number_of_iterations = static_cast<uint64_t>(std::ceil(vector_size / 16));
    constexpr uint8_t step_size = 16;
    auto current_sum_avx = _mm_set1_epi32(0);

    for (uint64_t i = 0; i < number_of_iterations; ++i) {
        auto vector_pointer = in_out_vector.data() + (i * step_size);
        auto load_part = _mm_loadu_si128(reinterpret_cast<const __m128i_u *>(vector_pointer));
        auto shifted_values = _mm_slli_si128(load_part, 1);

        auto summation = _mm_add_epi8(load_part, shifted_values);
        shifted_values = _mm_slli_si128(summation, 2);

        summation = _mm_add_epi8(summation, shifted_values);

        shifted_values = _mm_slli_si128(summation, 4);
        summation = _mm_add_epi8(summation, shifted_values);

        shifted_values = _mm_slli_si128(summation, 8);
        summation = _mm_add_epi8(summation, shifted_values);
        summation = _mm_add_epi8(summation, current_sum_avx);
        _mm_storeu_si128(reinterpret_cast<__m128i_u *>(vector_pointer), summation);
        current_sum_avx = _mm_set1_epi8(in_out_vector[i + step_size - 1]);

    }
    // Handle the last elements that are not processed
    for (uint64_t i = number_of_iterations * step_size; i < vector_size; ++i) {
        in_out_vector[i] += in_out_vector[i - 1];
    }

}

#endif //AVX2_PREFIX_SUM_AVX2_PREFIX_HPP
