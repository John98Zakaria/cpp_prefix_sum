//
// Created by jn98zk on 21.04.24.
//
#include "benchmark/benchmark.h"
#include "../pure_cpp_prefix_sum.hpp"

static void measure_simple_prefix_sort(benchmark::State &state) {
    std::size_t size = 1 << 20;
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        auto res = prefix_sum(numbers);
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(measure_simple_prefix_sort)->Unit(benchmark::kMicrosecond)->Iterations(500);


static void measure_cpp_prefix_sort(benchmark::State &state) {
    std::size_t size = state.range(0);
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);


    for (auto _: state) {
        std::vector<uint8_t> output(size);

        std::inclusive_scan(std::execution::par_unseq,
                            numbers.begin(),
                            numbers.end(),
                            output.begin(),
                            std::plus());
        benchmark::DoNotOptimize(output);
    }
}

BENCHMARK(measure_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);


static void measure_inplace_cpp_prefix_sort(benchmark::State &state) {
    std::size_t size = state.range(0);
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

static void measure_inplace_val_array_cpp_prefix_sort(benchmark::State &state) {
    std::size_t size = state.range(0);
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_val_array_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_val_array_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

// Run the benchmark
BENCHMARK_MAIN();
