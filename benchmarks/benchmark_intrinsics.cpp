//
// Created by jn98zk on 01.05.24.
//

#include "benchmark/benchmark.h"
#include "../intrinsics_prefix_sum.hpp"

template<std::integral T>
static void measure_inplace_sse2_t_prefix_sum(benchmark::State &state) {
    std::size_t size = state.range(0);
    std::vector<T> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_sse2_prefix_sum_t(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_sse2_t_prefix_sum<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);
BENCHMARK(measure_inplace_sse2_t_prefix_sum<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);
BENCHMARK(measure_inplace_sse2_t_prefix_sum<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);
BENCHMARK(measure_inplace_sse2_t_prefix_sum<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);


template<std::integral T>
static void measure_avx2_t_prefix_sort(benchmark::State &state) {
    std::size_t size = state.range(0);
    std::vector<T> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_avx2_prefix_sum_t(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_avx2_t_prefix_sort<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_avx2_t_prefix_sort<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_avx2_t_prefix_sort<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_avx2_t_prefix_sort<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

BENCHMARK_MAIN();