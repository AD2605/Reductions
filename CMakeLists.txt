cmake_minimum_required(VERSION 3.29)
project(Reductions LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(bench_reduce reduction.cu)

target_link_libraries(bench_reduce CUDA::cudart)