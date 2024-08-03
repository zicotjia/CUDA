cmake_minimum_required(VERSION 3.16)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(CUDA cuda_array_sum.cu

set_target_properties(CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
