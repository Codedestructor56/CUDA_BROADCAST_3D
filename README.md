# CUDA Broadcast Matrix Addition

This project demonstrates how to perform broadcast matrix addition(only 3D) using CUDA and PyTorch. The goal is to leverage GPU computation for efficient matrix operations in a simple example.

## Prerequisites

Before you start, make sure you have the following installed on your system:

- CUDA Toolkit
- PyTorch with CUDA support
- CMake (version 3.18 or higher)
- A C++ compiler (e.g., `g++`)

## Project Structure

The project consists of the following files and directories:

- `src/`
  - `cuda_kernel.cu`: Contains the CUDA kernel function and its launcher.
  - `cuda_app.cpp`: Contains the main application logic and PyTorch integration.
  - `cuda-kernel.h`: Header file for the CUDA kernel function.
- `include/`
  - `cuda-app.h`: Header file for the application.
- `CMakeLists.txt`: CMake build configuration file.

## Code Explanation

### CUDA Kernel

The CUDA kernel performs element-wise addition of two tensors, broadcasting them to a common shape. Here's the relevant code from `cuda_kernel.cu`:

```cpp
#include <stdio.h>
#include <c10/cuda/CUDAException.h>
#include "cuda-kernel.h"

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__ void broadcast_kernel_3d(float* a, float * b, float * out, int s1_0, int s1_1, int s1_2, int s2_0, int s2_1, int s2_2) {
    int i = blockIdx.z * blockDim.z + threadIdx.z; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > s1_0 || j > s1_1 || k > s1_2) {
        return;
    }

    float o = 0.0f;
    float a_value = a[(i % s1_0) * s1_1 * s1_2 + (j % s1_1) * s1_2 + (k % s1_2)];
    float b_value = b[(i % s2_0) * s2_1 * s2_2 + (j % s2_1) * s2_2 + (k % s2_2)];

    o = a_value + b_value;
    out[i * s1_1 * s1_2 + j * s1_2 + k] = o;
}
```


## Main Application

The main application integrates PyTorch with CUDA, creates tensors, and calls the broadcast kernel. Here's a relevant snippet from `cuda_app.cpp`:

```cpp
torch::Tensor broadcast_matrix(torch::Tensor a, torch::Tensor b) {
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
```




## CMake Configuration Explanation

Let's break down the `CMakeLists.txt` file line by line:

```cmake
```cmake_minimum_required(VERSION 3.18 FATAL_ERROR)```
This line sets the minimum required version of CMake to 3.18. If a version lower than 3.18 is used, CMake will produce a fatal error and stop.

```project(cuda-app LANGUAGES CXX CUDA)```
This line sets up the project named "cuda-app" with C++ and CUDA languages enabled.

```file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE CUDA_FILES CONFIGURE_DEPENDS src/*.cu)
```
These lines use the file command to search recursively for all .cpp and .cu files in the src directory and its subdirectories and store them in the variables CPP_FILES and CUDA_FILES, respectively.


```set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})```
Here, the host compiler for CUDA code is set to the C++ compiler.

```set_target_properties(cuda-app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75" 
)
```
This sets properties for the "cuda-app" target, enabling separable compilation for CUDA code and specifying the compute architecture to be targeted as "75"(you should set it to your own cuda sm arch).



## Compilation Steps

 - Create a build directory:

```mkdir build
cd build
```


 - Run CMake to configure the project:

```cmake ..```

 - Build the project using make:

```make```


## Running the Application
Once the project is built, you can run the application. If CUDA is available, the tensors will be created on the GPU; otherwise, they will be created on the CPU.

```./cuda-app```
The output will display the result of the broadcasted addition of two randomly generated tensors.

