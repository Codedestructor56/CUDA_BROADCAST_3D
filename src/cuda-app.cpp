#include <iostream>
#include <torch/torch.h>
#include "cuda-kernel.h"
#include "cuda-app.h"
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

std::vector<int64_t> get_dim(const std::vector<int64_t>& s1, const std::vector<int64_t>& s2) {
    std::vector<int64_t> broadcast_shape(3);
    for (size_t i = 0; i < 3; ++i) {
        broadcast_shape[i] = std::max(s1[i], s2[i]);
    }
    return broadcast_shape;
}

torch::Tensor initialize_broadcast_output(const std::vector<int64_t>& s1, const std::vector<int64_t>& s2) { 
    std::vector<int64_t> broadcast_shape = get_dim(s1, s2);
    torch::Tensor output = torch::zeros(broadcast_shape, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return output;
}

void broadcast_cuda(float* a, float* b, float* out, const std::vector<int>& s1, const std::vector<int>& s2, const std::vector<int>& broadcast_shape) {
    launch_broadcast_kernel_3d(a, b, out, s1, s2, broadcast_shape);
}

torch::Tensor broadcast_matrix(torch::Tensor a, torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    std::vector<int64_t> s1 = a.sizes().vec();
    std::vector<int64_t> s2 = b.sizes().vec();
    CHECK_BROADCASTABLE(s1, s2);

    torch::Tensor output = initialize_broadcast_output(s1, s2);
    std::vector<int64_t> broadcast_shape = get_dim(s1, s2);

    std::vector<int> s1_int(s1.begin(), s1.end());
    std::vector<int> s2_int(s2.begin(), s2.end());
    std::vector<int> broadcast_shape_int(broadcast_shape.begin(), broadcast_shape.end());

    if (std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end())) {
        broadcast_cuda(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), s1_int, s2_int, broadcast_shape_int);
    } else {
        broadcast_cuda(b.data_ptr<float>(), a.data_ptr<float>(), output.data_ptr<float>(), s2_int, s1_int, broadcast_shape_int);
    }

    return output;
}
