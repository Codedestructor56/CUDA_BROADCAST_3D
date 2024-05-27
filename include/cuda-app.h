#ifndef CUDA_APP_H
#define CUDA_APP_H
#include <torch/torch.h>

torch::Tensor broadcast_sum(torch::Tensor a, torch::Tensor b);


#endif

