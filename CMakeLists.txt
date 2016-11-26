cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Geigen CXX)
option(BUILD_PROTOTYPE "Build the prototype demo" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -Xcompiler -Wno-unused,-Wno-language-extension-token")

find_package(BLAS REQUIRED)

# Vendored dependencies.
# add_subdirectory(vendor/cudalicious)

include_directories(
  include
  vendor/cudalicious/include
  /usr/local/magma/include
)

link_directories(
  /usr/local/magma/lib
)

cuda_add_library(geigen
  src/geigen.cu
)

target_link_libraries(geigen
  ${BLAS_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  /usr/local/magma/lib/libmagma.a
)

if (BUILD_PROTOTYPE)
  add_executable(prototype src/prototype.cpp)
  target_link_libraries(prototype geigen)
endif()
