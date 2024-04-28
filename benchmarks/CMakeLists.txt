find_package(benchmark CONFIG REQUIRED)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark benchmark::benchmark CompilerFlags)