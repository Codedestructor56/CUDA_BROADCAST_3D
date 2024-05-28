#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_broadcast_kernel_3d(float* a, float* b, float* out, const std::vector<int>& s1, const std::vector<int>& s2, const std::vector<int>& broadcast_shape); 

#ifdef __cplusplus
}
#endif

#define THREADS 512

#endif
