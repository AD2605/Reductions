#ifndef DEFINES_HPP
#define DEFINES_HPP

#define FULL_MASK 0xffffffff
#define HALF_MASK 0xffff0000
#define QUARTER_MASK 0xff000000

#define checkCudaError(error_code) \
  { check_cuda_error((error_code), __FILE__, __LINE__); }

void check_cuda_error(cudaError error_code, const char *file, int line_number) {
  if (error_code != cudaError::cudaSuccess) {
    auto cuda_error_string = std::string(cudaGetErrorString(error_code)) +
                             " in file: " + std::string(file) +
                             " at line: " + std::to_string(line_number);
    throw std::runtime_error(cuda_error_string);
  }
}

constexpr std::size_t num_values = std::size_t(1) << 30;
constexpr int max_values_per_block = 4096;
constexpr int num_threads_per_block = 1024;
constexpr int max_blocks_per_sm = 2048 / num_threads_per_block;
constexpr int max_cluster_size = 8;  // 8 blocks per cluster limit
constexpr int max_values_per_cluster = max_cluster_size * max_values_per_block;

// Grid Dim, block Dim, Cluster Dim, Uses cluster launch
using launch_config_t = std::pair<std::tuple<dim3, dim3, dim3>, bool>;

#endif
