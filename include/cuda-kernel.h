#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>

#ifdef __cplusplus
extern "C" {
#endif

torch::Tensor broadcast_cuda(torch::Tensor a, torch::Tensor b);

#ifdef __cplusplus
}
#endif

#define THREADS 512

#endif
