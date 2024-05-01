//
// Created by jn98zk on 21.04.24.
//
#ifndef PURE_CPP_PREFIX_SUM_HPP
#define PURE_CPP_PREFIX_SUM_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <execution>
#include <immintrin.h>
#include <valarray>
#include <cstdint>
#include <vector>

/**
 * Compiler Vectorized Prefix Sum.
 *
 * This version is slow despite the vectorization due to the fact that the input array is read multiple times.
 *
 * @tparam T Vector Type
 * @param current_iteration_input The input vector which will be also used for temporary storage.
 * @return A new vector with the prefix sum of the input vector.
 */
template<typename T>
std::vector<T> prefix_sum(std::vector<T> current_iteration_input) {
    std::vector<T> next_iteration_input(current_iteration_input);

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = current_iteration_input.size();
    auto const number_of_iterations = static_cast<uint8_t>(std::ceil(std::log2(vector_size)));
    for (uint8_t iteration_number = 0; iteration_number < number_of_iterations; ++iteration_number) {

        // The number of elements that we need to copy.
        const uint64_t boundary_size = 1 << iteration_number;
        // The distance between the elements that will be summed to together in an interation.
        const uint64_t shift_size = boundary_size;

        // We create a copy of the first boundary_size elements
        for (uint64_t i = 0; i < boundary_size; ++i) {
            next_iteration_input[i] = current_iteration_input[i];
        }

        // Sum the elements that are have a distance of shift_size to each other.
        for (uint64_t i = 0; i < vector_size - boundary_size; ++i) {
            next_iteration_input[i + shift_size] = current_iteration_input[i] + current_iteration_input[i + shift_size];
        }

        // Swap the vectors for the next iteration.
        std::swap(current_iteration_input, next_iteration_input);
    }


    return current_iteration_input;
}


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
 * Inplace Prefix Sum using AVX2
 *
 * @param in_out_vector The vector which will be used for the prefix sum and overwritten with the result.
 */
void inplace_avx2_prefix_sum(std::vector<uint8_t> &in_out_vector) {

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = in_out_vector.size();
    // For some reason this is faster than incrementing i by 16.
    auto const number_of_iterations = static_cast<uint64_t>(std::ceil(vector_size / 32));
    constexpr uint8_t avx_step_size = 32;
    auto current_sum_avx = _mm256_set1_epi8(0);

    constexpr uint8_t avx_register_width = 16;

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    // For some reason this is faster than incrementing i by 16.

    for (uint64_t i = 0; i < number_of_iterations; ++i) {
        auto const vector_pointer = in_out_vector.data() + (i * avx_step_size);
        auto load_part = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(vector_pointer));

        auto right_shift = _mm256_slli_si256(load_part, 1);
        auto left_shift = _mm256_srli_si256(load_part, (avx_register_width - 1));
        auto merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        auto shifted_values = _mm256_or_si256(right_shift, merge_mask);

        auto summation = _mm256_add_epi8(load_part, shifted_values);
        right_shift = _mm256_slli_si256(summation, 2);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 2));

        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        summation = _mm256_add_epi8(summation, shifted_values);


        // Add write into the debug arrays

        right_shift = _mm256_slli_si256(summation, 4);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 4));
        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        summation = _mm256_add_epi8(summation, shifted_values);


        right_shift = _mm256_slli_si256(summation, 8);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 8));
        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        summation = _mm256_add_epi8(summation, shifted_values);


        right_shift = _mm256_slli_si256(summation, 16);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 16));
        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        summation = _mm256_add_epi8(summation, shifted_values);
        summation = _mm256_add_epi8(summation, current_sum_avx);
        _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(vector_pointer), summation);
        current_sum_avx = _mm256_set1_epi8(in_out_vector[(i + 1) * avx_step_size - 1]);

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
    std::vector<T> shift_view(avx_step_size);
    std::vector<T> sum_view(avx_step_size);
    for (uint64_t i = 0; i < number_of_iterations; ++i) {
        auto const vector_pointer = in_out_vector.data() + (i * avx_step_size);
        auto load_part = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(vector_pointer));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), load_part);

        auto right_shift = _mm256_slli_si256(load_part, 1 * unit_size);
        auto left_shift = _mm256_srli_si256(load_part, (avx_register_width - 1) * unit_size);
        auto merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        auto shifted_values = _mm256_or_si256(right_shift, merge_mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(shift_view.data()), shifted_values);

        auto summation = vector_add_function(load_part, shifted_values);
        // Store in sum_view for debugging
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), summation);
        right_shift = _mm256_slli_si256(summation, 2 * unit_size);
        left_shift = _mm256_srli_si256(summation, (avx_register_width - 2) * unit_size);

        merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
        shifted_values = _mm256_or_si256(right_shift, merge_mask);
        // Store in shift_view for debugging
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(shift_view.data()), shifted_values);
        summation = vector_add_function(summation, shifted_values);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), summation);


        // Add write into the debug arrays
        if constexpr (unit_size <= 4) {

            right_shift = _mm256_slli_si256(summation, 4 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 4) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(shift_view.data()), shifted_values);
            summation = vector_add_function(summation, shifted_values);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), summation);
        }


        if constexpr (unit_size <= 2) {
            right_shift = _mm256_slli_si256(summation, 8 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 8) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(shift_view.data()), shifted_values);
            summation = vector_add_function(summation, shifted_values);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), summation);
        }


        if constexpr (unit_size == 1) {
            right_shift = _mm256_slli_si256(summation, 16 * unit_size);
            left_shift = _mm256_srli_si256(summation, (avx_register_width - 16) * unit_size);
            merge_mask = _mm256_permute2x128_si256(left_shift, left_shift, 0b00001000);
            shifted_values = _mm256_or_si256(right_shift, merge_mask);
            // Store in shift_view for debugging
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(shift_view.data()), shifted_values);
            summation = vector_add_function(summation, shifted_values);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_view.data()), summation);
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

/**
 * Inplace Prefix Sum using SSE2
 * @param in_out_vector The vector which will be used for the prefix sum and overwritten with the result.
 */
void inplace_cpp_prefix_sum(std::vector<uint8_t> &in_out_vector) {

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = in_out_vector.size();
    // For some reason this is faster than incrementing iteration_limit by 16.
    constexpr uint8_t step_size = 16;
    std::array<uint8_t, step_size> avx_mock_1 = {};
    std::array<uint8_t, step_size> avx_mock_2 = {};
    uint8_t previous_sum = 0;
    auto const number_of_iterations = static_cast<uint8_t>(std::ceil(vector_size / step_size));


    for (uint64_t i = 0; i < number_of_iterations; i += 1) {
        auto vector_pointer = in_out_vector.data() + i * step_size;
        std::fill(avx_mock_1.begin(), avx_mock_1.end(), 0);
        std::fill(avx_mock_2.begin(), avx_mock_2.end(), 0);
        std::copy(vector_pointer, vector_pointer + step_size - 1, avx_mock_1.begin() + 1);

        for (int j = 0; j < step_size; ++j) {
            avx_mock_1[j] = avx_mock_1[j] + in_out_vector[i + j];
        }
        std::copy(avx_mock_1.begin(), avx_mock_1.end() - 2, avx_mock_2.begin() + 2);
        for (int j = 0; j < step_size; ++j) {
            avx_mock_1[j] = avx_mock_2[j] + avx_mock_1[j];
        }


        std::copy(avx_mock_1.begin(), avx_mock_1.end() - 4, avx_mock_2.begin() + 4);
        avx_mock_2[0] = 0;
        avx_mock_2[1] = 0;
        avx_mock_2[2] = 0;
        avx_mock_2[3] = 0;
        for (int j = 0; j < step_size; ++j) {
            avx_mock_1[j] = avx_mock_2[j] + avx_mock_1[j];
        }

        std::copy(avx_mock_1.begin(), avx_mock_1.end() - 8, avx_mock_2.begin() + 8);
        avx_mock_2[0] = 0;
        avx_mock_2[1] = 0;
        avx_mock_2[3] = 0;
        avx_mock_2[4] = 0;
        avx_mock_2[5] = 0;
        avx_mock_2[6] = 0;
        avx_mock_2[7] = 0;
        for (int j = 0; j < step_size; ++j) {
            avx_mock_1[j] = avx_mock_2[j] + avx_mock_1[j] + previous_sum;
        }

        previous_sum = avx_mock_1[17];

        std::copy(avx_mock_1.begin(), avx_mock_1.end(), vector_pointer);


    }
    // Handle the last elements that are not a multiple of 32,
    for (uint64_t i = number_of_iterations * step_size; i < vector_size; ++i) {
        in_out_vector[i] += in_out_vector[i - 1];
    }

}


/**
 * Inplace Prefix Sum using SSE2
 * @param in_out_vector The vector which will be used for the prefix sum and overwritten with the result.
 */
void inplace_val_array_cpp_prefix_sum(std::vector<uint8_t> &in_out_vector) {

    // A vector has at most vector_size of 2**64 so the max number of iterations (64) will fit here
    unsigned long vector_size = in_out_vector.size();
    // For some reason this is faster than incrementing iteration_limit by 16.
    constexpr uint8_t step_size = 16;
    std::valarray<uint8_t> avx_mock_1(step_size);
    std::valarray<uint8_t> avx_mock_2(step_size);
    std::valarray<uint8_t> previous_sum(step_size);

    auto const number_of_iterations = static_cast<uint8_t>(std::ceil(vector_size / step_size));

    for (uint64_t i = 0;
         i < number_of_iterations; i += 1) {
        auto vector_pointer = in_out_vector.data() + i * step_size;
        avx_mock_1 = 0;
        avx_mock_2 = 0;


        std::copy(vector_pointer, vector_pointer + step_size - 1, begin(avx_mock_1) + 1);

        for (int j = 0; j < step_size; ++j) {
            avx_mock_1[j] = avx_mock_1[j] + in_out_vector[i * step_size + j];
        }

        // This is twice as fast as the using the built-in shift for valarray.
        std::copy(begin(avx_mock_1), end(avx_mock_1) - 2, begin(avx_mock_2) + 2);
        avx_mock_1 = avx_mock_2 + avx_mock_1;


        std::copy(begin(avx_mock_1), end(avx_mock_1) - 4, begin(avx_mock_2) + 4);
        avx_mock_2[0] = 0;
        avx_mock_2[1] = 0;
        avx_mock_2[2] = 0;
        avx_mock_2[3] = 0;
        avx_mock_1 = avx_mock_2 + avx_mock_1;

        std::copy(begin(avx_mock_1), end(avx_mock_1) - 8, begin(avx_mock_2) + 8);
        avx_mock_2[0] = 0;
        avx_mock_2[1] = 0;
        avx_mock_2[3] = 0;
        avx_mock_2[4] = 0;
        avx_mock_2[5] = 0;
        avx_mock_2[6] = 0;
        avx_mock_2[7] = 0;

        avx_mock_1 = avx_mock_2 + avx_mock_1;

        avx_mock_1 = avx_mock_1 + previous_sum;

        previous_sum = avx_mock_1[17];

        std::copy(begin(avx_mock_1), end(avx_mock_1), vector_pointer);


    }
    // Handle the last elements that are not a multiple of 16,
    for (uint64_t i = number_of_iterations * step_size; i < vector_size; ++i) {
        in_out_vector[i] += in_out_vector[i - 1];
    }

}

#endif //PURE_CPP_PREFIX_SUM_HPP
