
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "grid_reduction.hpp"
#include "utils.hpp"

int main() {
  cudaDeviceProp properties;
  checkCudaError(cudaGetDeviceProperties(&properties, 0));
  const int max_blocks = properties.multiProcessorCount * max_blocks_per_sm;
  auto peak_theoritical_bandwidth =
      ((double)properties.memoryBusWidth * double(properties.memoryClockRate) * 2 * 1000) /
      (8 * 1e9);
  std::cout << "Peak Theoritical Bandwidth = " << peak_theoritical_bandwidth << std::endl;

  int compute_capability = properties.major;

  float *input;
  float *output;
  checkCudaError(cudaMalloc(&input, num_values * sizeof(float)));
  checkCudaError(cudaMalloc(&output, sizeof(float)));
  std::vector<float> input_host(num_values);
  populate_with_random(input_host.data(), num_values);

  checkCudaError(
      cudaMemcpy(input, input_host.data(), num_values * sizeof(float), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  cudaStream_t stream;
  checkCudaError(cudaStreamCreate(&stream));

  auto launch_config = get_launch_parameters(compute_capability, num_values, max_blocks);
  launch_kernel(launch_config, stream, input, output, num_values, launch_config.second);
  float host_sum = 0;

  for (const auto &v : input_host) host_sum += v;

  cudaDeviceSynchronize();

  float output_host = -1001;
  cudaMemcpy(&output_host, output, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Reduction Value = " << output_host << std::endl;
  std::cout << "Host calculated Value = " << host_sum << std::endl;
  if (relatively_equal(output_host, host_sum)) {
    constexpr int num_iterations = 100;
    benchmark_kernel(num_iterations, peak_theoritical_bandwidth, launch_config, stream, input,
                     output, num_values, launch_config.second);
  } else {
    throw std::runtime_error("Incorrect Reduction result");
  }
}
