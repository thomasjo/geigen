cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Geigen CXX)
set(CMAKE_CXX_STANDARD 14)

# Resolve CUDA dependency.
find_package(CUDA REQUIRED)

# Set CUDA/nvcc compilation flags.
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11")

# Add vendored dependencies.
# add_subdirectory(vendor/cudalicious)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cudalicious/include
)

cuda_add_executable(prototype
  src/prototype.cu
)

target_link_libraries(prototype
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
)
