cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda-app LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};../libtorch/share/cmake/")
find_package(Torch REQUIRED)
find_package(ATen REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Separate source files into C++ and CUDA files
file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE CUDA_FILES CONFIGURE_DEPENDS src/*.cu)

add_executable(cuda-app ${CPP_FILES} ${CUDA_FILES})
target_include_directories(cuda-app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cuda-app "${TORCH_LIBRARIES}" ${CUDAToolkit_LIBRARIES})

set_property(TARGET cuda-app PROPERTY CXX_STANDARD 17)

set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE CUDA)

# Ensure that NVCC uses the correct compiler for host code
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Set additional CUDA compilation flags if necessary
set_target_properties(cuda-app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75" 
)
