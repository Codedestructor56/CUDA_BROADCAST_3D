#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#include "cuda-kernel.h"

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

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

    o = a_value + b_value;
    out[i * s1_1 * s1_2 + j * s1_2 + k] = o;
}

void launch_broadcast_kernel_3d(float* a, float* b, float* out, const std::vector<int>& s1, const std::vector<int>& s2, const std::vector<int>& broadcast_shape) {
    dim3 tpb(8, 8, 8);
    dim3 blocks(cdiv(broadcast_shape[1], tpb.x), cdiv(broadcast_shape[2], tpb.y), cdiv(broadcast_shape[0], tpb.z));

    broadcast_kernel_3d<<<blocks, tpb>>>(
        a, b, out, 
        s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
