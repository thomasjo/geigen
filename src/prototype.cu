#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template<typename T>
void print_matrix(std::vector<T> matrix, int n)
{
  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      std::cout << std::setprecision(6) << std::setw(12) << matrix[IDX2C(i, j, n)];
    }
    std::cout << "\n";
  }
}

__device__
float get_reflector_coefficient(const float* source, const float* tau, const int n, const int reflector_index, const int row, const int col) {
  if (row < reflector_index || col < reflector_index) return (row == col) ? 1.0f : 0.0f;

  const auto row_coefficient = (row == reflector_index) ? 1.0f : source[reflector_index * n + row];
  const auto col_coefficient = (col == reflector_index) ? 1.0f : source[reflector_index * n + col];

  return ((row == col) ? 1.0f : 0.0f) - tau[reflector_index] * row_coefficient * col_coefficient;
}

__global__
void construct_q_matrix(float* q, const float* source, const float* tau, const int n)
{
  // Matrix is stored in column-major order, so x gives the row index, and y gives the column index.
  const auto row = blockDim.x * blockIdx.x + threadIdx.x;
  const auto col = blockDim.y * blockIdx.y + threadIdx.y;
  const auto idx = col * n + row;

  if (idx >= n * n) return;

  // TODO(thomasjo): Do this in a smarter manner.
  q[idx] = (row == col) ? 1.0f : 0.0f;
  __syncthreads();

  for (auto k = 0; k < n; ++k) {
    auto inner_product = 0.0f;

    for (auto i = 0; i < n; ++i) {
      const auto row_coefficient = q[i * n + row];
      const auto col_coefficient = get_reflector_coefficient(source, tau, n, k, i, col);

      inner_product += row_coefficient * col_coefficient;
    }

    q[idx] = inner_product;
  }
}

int main(int argc, char* argv[])
{
  const auto n = 4;
  const std::vector<float> matrix {
     5, -2, -1, 0,
    -2,  5,  0, 1,
    -1,  0,  5, 2,
     0,  1,  2, 5,
  };

  std::cout << "Input matrix:\n";
  print_matrix(matrix, n);

  // Create handles.
  cusolverDnHandle_t solver_handle = nullptr;
  auto solver_status = cusolverDnCreate(&solver_handle);
  assert(solver_status == CUSOLVER_STATUS_SUCCESS);

  cublasHandle_t blas_handle = nullptr;
  auto blas_status = cublasCreate(&blas_handle);
  assert(blas_status == CUBLAS_STATUS_SUCCESS);

  // Allocate device memory.
  float* dev_matrix;
  auto cuda_status = cudaMalloc(&dev_matrix, sizeof(float) * n * n);
  assert(cuda_status == cudaSuccess);

  float* dev_tau;
  cuda_status = cudaMalloc(&dev_tau, sizeof(float) * n);
  assert(cuda_status == cudaSuccess);

  int* dev_info;
  cuda_status = cudaMalloc(&dev_info, sizeof(int));
  assert(cuda_status == cudaSuccess);

  // Determine workspace size.
  int workspace_size;
  solver_status = cusolverDnSgeqrf_bufferSize(solver_handle, n, n, dev_matrix, n, &workspace_size);
  assert(solver_status == CUSOLVER_STATUS_SUCCESS);

  float* dev_workspace;
  cuda_status = cudaMalloc(&dev_workspace, sizeof(float) * workspace_size);
  assert(cuda_status == cudaSuccess);

  // Copy data to device memory.
  cuda_status = cudaMemcpy(dev_matrix, matrix.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);

  // Compute QR factorization.
  solver_status = cusolverDnSgeqrf(solver_handle, n, n, dev_matrix, n, dev_tau, dev_workspace, workspace_size, dev_info);
  assert(solver_status == CUSOLVER_STATUS_SUCCESS);

  int info;
  cuda_status = cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  assert(info == 0);

  const dim3 blocks(2, 2);
  const dim3 threads(2, 2);

  // Allocate device memory.
  float* dev_q;
  cuda_status = cudaMalloc(&dev_q, sizeof(float) * n * n);
  assert(cuda_status == cudaSuccess);

  // Initialize device matrix to unity.
  // cuda_status = cudaMemset(dev_q, 0.0f, sizeof(float) * n * n);
  // assert(cuda_status == cudaSuccess);

  construct_q_matrix<<<blocks, threads>>>(dev_q, dev_matrix, dev_tau, n);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  // Allocate device memory.
  float* dev_qq;
  cuda_status = cudaMalloc(&dev_qq, sizeof(float) * n * n);
  assert(cuda_status == cudaSuccess);

  // Copy data to device memory.
  cuda_status = cudaMemcpy(dev_matrix, matrix.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);

  // --
  // Simulate a single iteration of traditional QR algorithm.
  const auto alpha = 1.0f;
  const auto beta = 0.0f;
  blas_status = cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, dev_q, n, dev_matrix, n, &beta, dev_qq, n);
  assert(blas_status == CUBLAS_STATUS_SUCCESS);

  blas_status = cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_qq, n, dev_q, n, &beta, dev_qq, n);
  assert(blas_status == CUBLAS_STATUS_SUCCESS);

  std::vector<float> result(n * n);
  cuda_status = cudaMemcpy(result.data(), dev_qq, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "\nResult:\n";
  print_matrix(result, n);
  // --

  std::vector<float> tau(n);
  cuda_status = cudaMemcpy(tau.data(), dev_tau, sizeof(float) * n, cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "\nTAU: ";
  for (const auto& t : tau) std::cout << t << ", ";
  std::cout << "\n";

  if (dev_qq)        cudaFree(dev_qq);
  if (dev_q)         cudaFree(dev_q);
  if (dev_matrix)    cudaFree(dev_matrix);
  if (dev_tau)       cudaFree(dev_tau);
  if (dev_workspace) cudaFree(dev_workspace);
  if (dev_info)      cudaFree(dev_info);

  if (solver_handle) cusolverDnDestroy(solver_handle);
  if (blas_handle)   cublasDestroy(blas_handle);

  cudaDeviceReset();

  return 0;
}
