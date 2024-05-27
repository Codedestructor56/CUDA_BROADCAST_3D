#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include "cuda-kernel.h"

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

// Kernel function
__global__ void broadcast_kernel_3d(float* a, float * b, float * out, int s1_0, int s1_1, int s1_2, int s2_0, int s2_1, int s2_2) {
    int i = blockIdx.z * blockDim.z + threadIdx.z; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= s1_0 || j >= s1_1 || k >= s1_2) {
        return;
    }

    float o = 0.0f;
    float a_value = a[(i % s1_0) * s1_1 * s1_2 + (j % s1_1) * s1_2 + (k % s1_2)];
    float b_value = b[(i % s2_0) * s2_1 * s2_2 + (j % s2_1) * s2_2 + (k % s2_2)];

    o = (float)(a_value + b_value);
    out[i * s1_1 * s1_2 + j * s1_2 + k] = o;
}

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

torch::Tensor broadcast_cuda(torch::Tensor a , torch::Tensor b) {
    std::vector<int64_t> s1 = a.sizes().vec();
    std::vector<int64_t> s2 = b.sizes().vec();

    if (s1.size() != s2.size()) {
        fprintf(stderr, "Error: Tensors do not have the same shape.\n");
        exit(-1);
    }

    if (s1.size() < 3 || s2.size() < 3) {
        fprintf(stderr, "Error: Tensors must have at least 3 dimensions.\n");
        exit(-1);
    }

    torch::Tensor output = initialize_broadcast_output(s1, s2);
    std::vector<int64_t> broadcast_shape = get_dim(s1, s2);

    dim3 tpb(8, 8, 8);
    dim3 blocks(cdiv(broadcast_shape[1], tpb.x), cdiv(broadcast_shape[2], tpb.y), cdiv(broadcast_shape[0], tpb.z));

    broadcast_kernel_3d<<<blocks, tpb>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), 
        s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return output;
}

