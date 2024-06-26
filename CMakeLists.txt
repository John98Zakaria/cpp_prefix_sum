cmake_minimum_required(VERSION 3.28)
project(avx2_prefix_sum)

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

MESSAGE(INFO ${CMAKE_MODULE_PATH})

include(CompilerWarnings)
include(EnableIntrinsics)
include(EnableVectorization)

add_library(CompilerFlags INTERFACE)

enable_compiler_warnings(CompilerFlags "OFF" "" "" "" "")
enable_intrinsics(CompilerFlags true)
enable_vectorization(CompilerFlags true)

add_executable(avx2_prefix main.cpp)
target_link_libraries(avx2_prefix CompilerFlags)

add_executable(avx2_shift avx2_shift.cpp)
target_link_libraries(avx2_shift CompilerFlags)

add_subdirectory(benchmarks)