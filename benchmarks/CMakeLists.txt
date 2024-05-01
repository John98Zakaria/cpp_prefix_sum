find_package(benchmark CONFIG REQUIRED)

add_executable(benchmark benchmark_pure_cpp.cpp)
target_link_libraries(benchmark benchmark::benchmark CompilerFlags OpenMP::OpenMP_CXX)
add_executable(benchmark_intrinsics benchmark_intrinsics.cpp)
target_link_libraries(benchmark_intrinsics benchmark::benchmark CompilerFlags OpenMP::OpenMP_CXX)
