//
// Created by jn98zk on 21.04.24.
//
#include "benchmark/benchmark.h"
#include "../pure_cpp_prefix_sum.hpp"

static void measure_simple_prefix_sort(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<uint8_t> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        auto res = prefix_sum(numbers);
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(measure_simple_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);


template <std::integral T>
static void measure_cpp_prefix_sort(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<T> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);


    for (auto _: state) {
        std::inclusive_scan(std::execution::par_unseq,
                            numbers.begin(),
                            numbers.end(),
                            numbers.begin(),
                            std::plus());
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_cpp_prefix_sort<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_prefix_sort<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_prefix_sort<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_prefix_sort<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);


static void measure_inplace_cpp_prefix_sort(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<uint8_t> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

static void measure_inplace_val_array_cpp_prefix_sort(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<uint8_t> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_val_array_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_val_array_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

// Run the benchmark
BENCHMARK_MAIN();
