cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Geigen CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -Xcompiler -Wno-unused,-Wno-language-extension-token")

# Vendored dependencies.
# add_subdirectory(vendor/cudalicious)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cudalicious/include
)

cuda_add_library(geigen
  src/geigen.cu
)

target_link_libraries(geigen
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
)

add_executable(prototype
  src/prototype.cpp
)

target_link_libraries(prototype
  geigen
)
