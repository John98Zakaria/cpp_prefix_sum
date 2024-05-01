//
// Created by jn98zk on 21.04.24.
//
#ifndef INTRINSICS_PREFIX_SUM_HPP
#define INTRINSICS_PREFIX_SUM_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <execution>
#include <immintrin.h>
#include <valarray>
#include <cstdint>
#include <vector>


/**
 * Inplace Prefix Sum using SSE2 that supports different types up to 64 bits.
 *
 * @param in_out_vector The vector which will be used for the prefix sum and overwritten with the result.
 */
template<std::integral T>
constexpr void inplace_sse2_prefix_sum_t(std::vector<T> &in_out_vector) {


    constexpr uint8_t unit_size = sizeof(T);

    constexpr auto avx_step_size = []() -> auto {
        if constexpr (unit_size == 1) {
            return 16;
        } else if constexpr (unit_size == 2) {
            return 8;
        } else if constexpr (unit_size == 4) {
            return 4;
        } else {
            return 2;
        }
    }();

    constexpr auto vector_add_function = []() -> auto {
        if constexpr (unit_size == 1) {
            return _mm_add_epi8;
        } else if constexpr (unit_size == 2) {
            return _mm_add_epi16;
        } else if constexpr (unit_size == 4) {
            return _mm_add_epi32;
        } else {
            return _mm_add_epi64;
        }
    }();

    constexpr auto vector_set_function = []() -> auto {
        if constexpr (unit_size == 1) {
            return _mm_set1_epi8;
        } else if constexpr (unit_size == 2) {
            return _mm_set1_epi16;
        } else if constexpr (unit_size == 4) {
            return _mm_set1_epi32;
        } else {
            return _mm_set1_epi64x;
        }
    }();


    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    const std::size_t vector_size = in_out_vector.size();
    // For some reason this is faster than incrementing i by 16.
    auto const number_of_iterations = static_cast<uint64_t>(std::ceil(vector_size / avx_step_size));
    auto current_sum_avx = vector_set_function(T(0));
    for (uint64_t i = 0; i < number_of_iterations; ++i) {
        auto const vector_pointer = in_out_vector.data() + (i * avx_step_size);
        auto load_part = _mm_loadu_si128(reinterpret_cast<const __m128i_u *>(vector_pointer));

        auto shifted_values = _mm_slli_si128(load_part, 1 * unit_size);

        auto summation = vector_add_function(load_part, shifted_values);
        if constexpr (unit_size <= 4) {
            shifted_values = _mm_slli_si128(summation, 2 * unit_size);
            summation = vector_add_function(summation, shifted_values);
        }


        if constexpr (unit_size <= 2) {
            shifted_values = _mm_slli_si128(summation, 4 * unit_size);
            summation = vector_add_function(summation, shifted_values);
        }


        if constexpr (unit_size == 1) {
            shifted_values = _mm_slli_si128(summation, 8 * unit_size);
            summation = vector_add_function(summation, shifted_values);
        }
        summation = vector_add_function(summation, current_sum_avx);
        _mm_storeu_si128(reinterpret_cast<__m128i_u *>(vector_pointer), summation);
        current_sum_avx = vector_set_function(in_out_vector[(i + 1) * avx_step_size - 1]);

    }
    // Handle the last elements that are not a multiple of 16,
    for (uint64_t i = number_of_iterations * avx_step_size; i < vector_size; ++i) {
        in_out_vector[i] += in_out_vector[i - 1];
    }

}


/**
 * Inplace Prefix Sum using AVX2 that supports different types up to 64 bits.
 *
 * @param in_out_vector The vector which will be used for the prefix sum and overwritten with the result.
 */
template<std::integral T>
constexpr void inplace_avx2_prefix_sum_t(std::vector<T> &in_out_vector) {


    constexpr uint8_t unit_size = sizeof(T);

    constexpr auto avx_step_size = []() -> auto {
        if constexpr (unit_size == 1) {
            return 32;
        } else if constexpr (unit_size == 2) {
            return 16;
        } else if constexpr (unit_size == 4) {
            return 8;
        } else {
            return 4;
        }
    }();

    constexpr auto vector_add_function = []() -> auto {
        if constexpr (unit_size == 1) {
            return _mm256_add_epi8;
        } else if constexpr (unit_size == 2) {
            return _mm256_add_epi16;
        } else if constexpr (unit_size == 4) {
            return _mm256_add_epi32;
        } else {
            return _mm256_add_epi64;
        }
    }();

    constexpr auto vector_set_function = []() -> auto {
        if constexpr (unit_size == 1) {
            return _mm256_set1_epi8;
        } else if constexpr (unit_size == 2) {
            return _mm256_set1_epi16;
        } else if constexpr (unit_size == 4) {
            return _mm256_set1_epi32;
        } else {
            return _mm256_set1_epi64x;
        }
    }();

    constexpr uint8_t avx_register_width = avx_step_size / 2;

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    const std::size_t vector_size = in_out_vector.size();
    // For some reason this is faster than incrementing i by 16.
    auto const number_of_iterations = static_cast<uint64_t>(std::ceil(vector_size / avx_step_size));
    auto current_sum_avx = vector_set_function(T(0));
    for (uint64_t i = 0; i < number_of_iterations; ++i) {
        auto const vector_pointer = in_out_vector.data() + (i * avx_step_size);
        auto load_part = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(vector_pointer));

        auto right_shift = _mm256_slli_si256(load_part, 1 * unit_size);
        auto left_shift = _mm256_srli_si256(load_part, (avx_register_width - 1) * unit_size);
        auto merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        auto shifted_values = _mm256_or_si256(right_shift, merge_mask);

        auto summation = vector_add_function(load_part, shifted_values);
        // Store in sum_view for debugging
        right_shift = _mm256_slli_si256(summation, 2 * unit_size);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 2) * unit_size);

        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        // Store in shift_view for debugging
        summation = vector_add_function(summation, shifted_values);


        // Add write into the debug arrays
        if constexpr (unit_size <= 4) {

            right_shift = _mm256_slli_si256(summation, 4 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 4) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            summation = vector_add_function(summation, shifted_values);
        }


        if constexpr (unit_size <= 2) {
            right_shift = _mm256_slli_si256(summation, 8 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 8) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            summation = vector_add_function(summation, shifted_values);
        }


        if constexpr (unit_size == 1) {
            right_shift = _mm256_slli_si256(summation, 16 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 16) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            summation = vector_add_function(summation, shifted_values);
        }
        summation = vector_add_function(summation, current_sum_avx);
        _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(vector_pointer), summation);
        current_sum_avx = vector_set_function(in_out_vector[(i + 1) * avx_step_size - 1]);

    }
    // Handle the last elements that are not a multiple of 16,
    for (uint64_t i = number_of_iterations * avx_step_size; i < vector_size; ++i) {
        in_out_vector[i] += in_out_vector[i - 1];
    }

}


#endif //INTRINSICS_PREFIX_SUM_HPP
