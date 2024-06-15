#ifndef UTILS_HPP
#define UITLS_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "defines.hpp"
#include "grid_reduction.hpp"

launch_config_t get_launch_parameters(int compute_capability, std::size_t num_elements,
                                      int max_blocks) {
  if (compute_capability >= 9) {
    if (num_elements > max_values_per_cluster) {
      auto num_clusters = (num_elements + max_values_per_cluster - 1) / max_values_per_cluster;
      auto num_blocks = num_clusters * max_cluster_size;
      num_blocks = num_blocks > max_blocks ? max_blocks : num_blocks;
      num_blocks = ((num_blocks + max_cluster_size - 1) / max_cluster_size) * max_cluster_size;
      return std::make_pair(std::tuple(dim3(num_blocks, 1, 1), dim3(num_threads_per_block, 1, 1),
                                       dim3(max_cluster_size, 1, 1)),
                            true);
    } else {
      int num_blocks = (num_values + max_values_per_block - 1) / max_values_per_block;
      num_blocks = num_blocks > max_blocks ? max_blocks : num_blocks;
      return std::make_pair(
          std::tuple(dim3(num_blocks, 1, 1), dim3(num_threads_per_block, 1, 1), dim3(1, 1, 1)),
          false);
    }
  } else {
    int num_blocks = (num_values + max_values_per_block - 1) / max_values_per_block;
    num_blocks = num_blocks > max_blocks ? max_blocks : num_blocks;
    return std::make_pair(
        std::tuple(dim3(num_blocks, 1, 1), dim3(num_threads_per_block, 1, 1), dim3(1, 1, 1)),
        false);
  }
}

template <typename... Args>
inline void launch_kernel(const launch_config_t &launch_config, cudaStream_t stream, Args... args) {
  auto uses_cluster_launch = launch_config.second;
  if (uses_cluster_launch) {
    cudaLaunchConfig_t config = {0};
    config.gridDim = std::get<0>(launch_config.first);
    config.blockDim = std::get<1>(launch_config.first);
    config.stream = stream;
    config.dynamicSmemBytes = 32;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = std::get<2>(launch_config.first).x;
    attribute[0].val.clusterDim.y = std::get<2>(launch_config.first).y;
    attribute[0].val.clusterDim.z = std::get<2>(launch_config.first).z;
    config.attrs = attribute;
    config.numAttrs = 1;
    checkCudaError(cudaLaunchKernelEx(&config, grid_reduce, args...));
  } else {
    auto grid_dim = std::get<0>(launch_config.first);
    auto block_dim = std::get<1>(launch_config.first);
    grid_reduce<<<grid_dim, block_dim, 32, stream>>>(args...);
  }
}

void populate_with_random(float *in, std::size_t size, float lowerLimit = -1.0f,
                          float higherLimit = 1.0f) {
  using engine = std::ranlux48_base;
  engine algo(0);
  std::uniform_real_distribution<float> distribution(lowerLimit, higherLimit);
  for (std::size_t i = 0; i < size; i++) {
    in[i] = float(distribution(algo));
  }
}

template <typename... Args>
void benchmark_kernel(int num_iterations, float theoritical_throughput,
                      const launch_config_t &launch_config, cudaStream_t stream, Args... args) {
  printf("Beginning benchmark with %d iterations \n", num_iterations);
  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start));
  checkCudaError(cudaEventCreate(&stop));
  float time = 0;
  for (int i = 0; i < num_iterations; i++) {
    float temp_t = 0;
    checkCudaError(cudaEventRecord(start, stream));
    launch_kernel(launch_config, stream, args...);
    checkCudaError(cudaEventRecord(stop, stream));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&temp_t, start, stop));
    time += temp_t;
  }
  double achieved_throughput =
      (num_values * sizeof(float) * 1e-9) / ((time / num_iterations) * 1e-3);
  std::cout << "Average time = " << time / num_iterations << std::endl;
  std::cout << "GBPS = " << achieved_throughput << std::endl;
  std::cout << "Achieved Throughput Percentage = "
            << (achieved_throughput / theoritical_throughput) * 100 << std::endl;
}

// https://floating-point-gui.de/errors/comparison/
bool relatively_equal(float a, float b, float epsilon = 1e-3) {
  float abs_a = std::abs(a);
  float abs_b = std::abs(b);
  float diff = std::abs(a - b);
  if (a == b) {
    return true;
  }
  auto float_min = 1.17549435E-38f;
  auto float_max = 0x1.fffffep127f;
  if (a == 0 || b == 0 || ((abs_a + abs_b) < float_min)) {
    return diff < (diff * epsilon);
  } else {
    return (diff / std::min(abs_a + abs_b, float_max)) < epsilon;
  }
}

#endif
