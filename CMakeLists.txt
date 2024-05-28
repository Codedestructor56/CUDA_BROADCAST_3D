cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda-app LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};../libtorch/share/cmake/")
find_package(Torch REQUIRED)
find_package(ATen REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)
add_executable(cuda-app ${SRC_FILES})
target_include_directories(cuda-app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cuda-app "${TORCH_LIBRARIES}" ${CUDAToolkit_LIBRARIES})
set_property(TARGET cuda-app PROPERTY CXX_STANDARD 17)

set_property(SOURCE ${SRC_FILES} PROPERTY LANGUAGE CUDA)

