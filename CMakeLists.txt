# Brendan Niebruegge
# Peter Dirks
# Homework 6
# CMakeLists.txt
# May 1, 2016

cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Werror -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# not sure what this does, but it keeps the CXX flags from breaking everything
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_executable( hw6 src/hw6.cpp src/util.cu include/sdkHelper.h include/util.cuh )
