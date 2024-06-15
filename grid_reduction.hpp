/**
 * @file grid_reduction.hpp
 * @author Atharva
 * @brief
 * @version 0.1
 * @date 2024-06-15
 *
 * @copyright Copyright (c) 2024 Atharva
 *
 */

#ifndef GRID_REDUCTION_HPP
#define GRID_REDUCTION_HPP

#include <cooperative_groups.h>

#include "defines.hpp"

// Each thread in a block loads 4 floats, reduces them, and then
// stores them in shared memory. Each block then reduces
// the values stored in the shared memory. Then the leader of
// each block atomically writes to the output.
// Each block reduces up to 4096 values.
// for now, number of elements are always a multiple of 4

// The H100 supports a cluster size of 8.
// Hence each cluster can reduce up to 8 * 4096 = 32768 values.
// All the blocks in cluster atomically add their reduced value to cluster
// leader Then each cluster leader atomically writes to global memory

template <int N = 16>  // Default to full warp reduction
__device__ __forceinline__ void warp_reduce(float &val, int mask = FULL_MASK) {
#pragma unroll(N)
  for (int offset = N; offset > 0; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset);
  }
}

__device__ __forceinline__ void cluster_reduce(float *smem, float *output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  namespace cg = cooperative_groups;
  auto cluster_group = cg::this_cluster();

  // Make sure each workgroup in the cluster is alive
  cluster_group.sync();
  // All the other clusters atomically add to cluster leader
  float val = 0;
  if (cluster_group.block_rank() == 0 && (threadIdx.x / 32 == 0)) {
    // A single warp gathers all the values
    if (threadIdx.x < max_cluster_size) {
      val = cluster_group.map_shared_rank(smem, threadIdx.x)[0];
    }
    __syncwarp();
    // quarter warp reduction
    warp_reduce<max_cluster_size / 2>(val, QUARTER_MASK);
  }
  cluster_group.sync();
  if (cluster_group.block_rank() == 0 && threadIdx.x == 0) {
    atomicAdd(output, val);
  }
#endif
}

// Output needs to be initialized with 0 or any other value desired

__global__ void grid_reduce(const float *input, float *out, std::size_t num_elements,
                            bool uses_cluster_reduce = false) {
  __shared__ float smem[32];
  auto warp_id = threadIdx.x / 32;
  auto is_warp_leader = (threadIdx.x % 32 == 0);
  if (is_warp_leader) {
    smem[warp_id] = 0;
  }

  float summed_value = 0;
  auto id = threadIdx.x + blockDim.x * blockIdx.x;
  // This loop basically has the same throughput as that of a copy kernel.
  for (; id < num_elements / 4; id += blockDim.x * gridDim.x) {
    float4 loaded_values = *reinterpret_cast<const float4 *>(input + id * 4);
    // Omitting overflow check for now

    // Reduction at register level
    summed_value += loaded_values.w + loaded_values.x + loaded_values.y + loaded_values.z;
  }
  // Now each warp can reduce the summed_value stored in register, (there are 32
  // warps per block) Reduction at warp level across the workgroup
  warp_reduce(summed_value);
  // There are 1024 threads in a block, warp id will span from 0 - 31
  if (is_warp_leader) {
    smem[warp_id] = summed_value;
  }
  // Finish partial workgroup level reduction
  __syncthreads();

  // Now single warp reduces the remaining 32 values
  if (warp_id == 0) {
    // Reusing summed_value
    summed_value = smem[threadIdx.x];
    // Final reduction being done by a single warp for that particular workgroup
    warp_reduce(summed_value);
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (uses_cluster_reduce) {
    cluster_reduce(smem, out);
  } else {
    if (threadIdx.x == 0) {
      atomicAdd(out, summed_value);
    }
  }
#else
  // Now block leader atomicAdds to global out. This is the costliest step and
  // further can be optimized via writing to scratchpad and then reducing it
  // futher, but that's left for later
  if (threadIdx.x == 0) {
    atomicAdd(out, summed_value);
  }
#endif
}

#endif
