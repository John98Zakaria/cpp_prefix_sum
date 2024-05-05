//
// Created by jn98zk on 21.04.24.
//
#include "benchmark/benchmark.h"
#include "parallel/numeric"
#include "../pure_cpp_prefix_sum.hpp"

template<std::integral T>
static void measure_simple_prefix_sum(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<T> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        auto res = prefix_sum(numbers);
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(measure_simple_prefix_sum<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_simple_prefix_sum<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_simple_prefix_sum<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_simple_prefix_sum<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

template<std::integral T>
static void measure_cpp_partial_sum(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<T> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);


    for (auto _: state) {
        std::partial_sum(numbers.begin(),
                         numbers.end(),
                         numbers.begin(),
                         std::plus());
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_cpp_partial_sum<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);


template<std::integral T>
static void measure_gnu_parallel_partial_sum(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<T> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);


    for (auto _: state) {
        __gnu_parallel::partial_sum(
                numbers.begin(),
                numbers.end(),
                numbers.begin(),
                std::plus());
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_gnu_parallel_partial_sum<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_gnu_parallel_partial_sum<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);
BENCHMARK(measure_gnu_parallel_partial_sum<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);
BENCHMARK(measure_gnu_parallel_partial_sum<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(
        500);


template<std::integral T>
static void measure_cpp_inplace_sum(benchmark::State &state) {
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


BENCHMARK(measure_cpp_partial_sum<uint8_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint16_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint32_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);
BENCHMARK(measure_cpp_partial_sum<uint64_t>)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);


static void measure_inplace_cpp_prefix_sum(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<uint8_t> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_cpp_prefix_sum)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

static void measure_inplace_val_array_cpp_prefix_sum(benchmark::State &state) {
    auto const size = state.range(0);
    std::vector<uint8_t> numbers(static_cast<std::size_t>(size));
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _: state) {
        inplace_val_array_cpp_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_val_array_cpp_prefix_sum)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 20)->Iterations(500);

// Run the benchmark
BENCHMARK_MAIN();
