#include <iostream>
#include <torch/torch.h>
#include "cuda-kernel.h"
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_BROADCASTABLE(s1, s2) \
    do { \
        if (s1.size() != s2.size()) { \
            fprintf(stderr, "Error: Tensors do not have the same number of dimensions.\n"); \
            exit(-1); \
        } \
        for (size_t i = 0; i < s1.size(); ++i) { \
            if (s1[i] != 1 && s2[i] != 1 && s1[i] != s2[i]) { \
                fprintf(stderr, "Error: Tensors are not broadcastable.\n"); \
                exit(-1); \
            } \
        } \
    } while(0)

def broadcast_matrix(torch::Tensor a, torch::Tensor b){
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_BROADCASTABLE(a,b);
  std::vector<int64_t> s1 = a.sizes().vec();
  std::vector<int64_t> s2 = b.sizes().vec();
  bool s1_is_larger = false;
  bool s2_is_larger = false;
  for (size_t i = 0; i < s1.size(); ++i) {
      if (s1[i] > s2[i]) {
          s1_is_larger = true;
      } else if (s1[i] < s2[i]) {
          s2_is_larger = true;
      }
  }
  if(s1_is_larger){
    broadcast_cuda(a,b);
  }
  else{
    broadcast_cuda(b,a);
  }
}
