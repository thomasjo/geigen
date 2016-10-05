cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Geigen CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")

# Support Xcode 7.3 side-by-side support...
if(EXISTS "$ENV{XCODE73_DIR}")
  set(MACOS_SDK_DIR "$ENV{XCODE73_DIR}/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk")
  include_directories(BEFORE SYSTEM
    ${MACOS_SDK_DIR}/usr/include
  )
endif()

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -Xcompiler -Wno-unused,-Wno-language-extension-token")

find_package(BLAS REQUIRED)

# Vendored dependencies.
# add_subdirectory(vendor/cudalicious)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cudalicious/include
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

add_executable(prototype
  src/prototype.cpp
)

target_link_libraries(prototype
  geigen
)
