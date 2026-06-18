cmake_minimum_required(VERSION 3.18)
project(fast_weights LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(BENCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# The dependency-free bench sources, compiled here under the Torch ABI flags.
add_library(bench_core STATIC
  ${BENCH_ROOT}/src/corpus.cpp
  ${BENCH_ROOT}/src/metric.cpp
  ${BENCH_ROOT}/src/diagnostics.cpp
  ${BENCH_ROOT}/src/experiment.cpp
  ${BENCH_ROOT}/models/context_model.cpp)
target_include_directories(bench_core PUBLIC ${BENCH_ROOT}/include)

add_executable(fw_test fw_test.cpp fast_weights_model.cpp)
target_include_directories(fw_test PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/tests ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fw_test bench_core "${TORCH_LIBRARIES}")
