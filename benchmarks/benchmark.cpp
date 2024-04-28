//
// Created by jn98zk on 21.04.24.
//
#include "benchmark/benchmark.h"
#include "../avx2_prefix.hpp"
#include <algorithm>
#include "../alligned_alloc.hpp"

typedef AlignedAllocator<int, 64> IntAligin ;
typedef AlignedAllocator<uint8_t, 64> UIntAligin ;
static void measure_avx_prefix_sort(benchmark::State &state) {
    std::size_t size = 1 << 20;
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _ : state) {
        auto res = prefix_sum(numbers);
        benchmark::DoNotOptimize(res);
    }
}


// Register the functions as a benchmark
BENCHMARK(measure_avx_prefix_sort)->Unit(benchmark::kMicrosecond)->Iterations(200);


static void measure_cpp_prefix_sort(benchmark::State &state) {
    std::size_t size = 1 << 20;
    std::vector<uint8_t,UIntAligin> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);



    for (auto _ : state) {
        std::vector<uint8_t> output(size);

        std::inclusive_scan(std::execution::seq,
                            numbers.begin(),
                            numbers.end(),
                            output.begin(),
                            std::plus());
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(measure_cpp_prefix_sort)->Unit(benchmark::kMicrosecond)->Iterations(200);

// Define another benchmark
static void measure_inplace_prefix_sort(benchmark::State &state) {
    std::size_t size = 1 << 20;
    std::vector<uint8_t> numbers(size);
    std::iota(numbers.begin(), numbers.end(), 0);

    for (auto _ : state) {
        inplace_prefix_sum(numbers);
        benchmark::DoNotOptimize(numbers);
    }
}

BENCHMARK(measure_inplace_prefix_sort)->Unit(benchmark::kMicrosecond)->Iterations(200);


// Run the benchmark
BENCHMARK_MAIN();
