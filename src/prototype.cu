#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cudalicious/cudalicious.h>

constexpr auto K_MAX = 50;

template<typename T>
void print_matrix(const std::vector<T>& matrix, const int n)
{
  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      const auto idx = j * n + i;
      std::cout << std::setprecision(6) << std::setw(12) << matrix[idx];
    }
    std::cout << "\n";
  }
}

template<typename T>
void print_device_matrix(T* dev_ptr, int n)
{
  std::vector<float> v(n * n);
  cuda::copy_to_host(v, dev_ptr);
  print_matrix(v, n);
}

__device__
float get_reflector_coefficient(const float* source, const float* tau, const int n, const int reflector_index, const int row, const int col)
{
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

int main()
{
  const auto n = 4;
  const std::vector<float> matrix {
     5, -2, -1, 0,
    -2,  5,  0, 1,
    -1,  0,  5, 2,
     0,  1,  2, 5,
  };
  assert(matrix.size() == n * n);

  std::cout << "Input matrix:\n";
  print_matrix(matrix, n);

  const std::vector<float> identity {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  };
  assert(identity.size() == n * n);

  // Create handles.
  cusolverDnHandle_t solver_handle = nullptr;
  auto solver_status = cusolverDnCreate(&solver_handle);
  assert(solver_status == CUSOLVER_STATUS_SUCCESS);

  cublasHandle_t blas_handle = nullptr;
  auto blas_status = cublasCreate(&blas_handle);
  assert(blas_status == CUBLAS_STATUS_SUCCESS);

  // Allocate device memory.
  auto dev_matrix = cuda::copy_to_device(matrix);

  // Determine workspace size.
  int workspace_size;
  solver_status = cusolverDnSgeqrf_bufferSize(solver_handle, n, n, dev_matrix, n, &workspace_size);
  assert(solver_status == CUSOLVER_STATUS_SUCCESS);

  auto dev_qr = cuda::allocate<float>(matrix.size());
  auto dev_tau = cuda::allocate<float>(n);
  auto dev_workspace = cuda::allocate<float>(workspace_size);
  auto dev_info = cuda::allocate<int>(1);

  auto dev_q = cuda::copy_to_device(identity);
  auto dev_eigvecs = cuda::copy_on_device(dev_q, matrix.size());

  cuda::copy_on_device(dev_qr, dev_matrix, matrix.size());

  for (auto k = 0; k < K_MAX; ++k) {
    // Compute QR factorization.
    solver_status = cusolverDnSgeqrf(solver_handle, n, n, dev_qr, n, dev_tau, dev_workspace, workspace_size, dev_info);
    assert(solver_status == CUSOLVER_STATUS_SUCCESS);

    int info;
    cuda::copy_to_host(&info, dev_info, 1);
    assert(info == 0);

    cuda::copy_to_device(dev_q, identity.data(), matrix.size());

    const dim3 blocks(2, 2);
    const dim3 threads(2, 2);
    construct_q_matrix<<<blocks, threads>>>(dev_q, dev_qr, dev_tau, n);
    cuda::device_sync();

    constexpr auto alpha = 1.0f;
    constexpr auto beta = 0.0f;

    // Compute A_k = Q_k^T * A_(k-1) * Q_k --> A_k converges to eigenvalues of A_0.
    blas_status = cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, dev_q, n, dev_matrix, n, &beta, dev_qr, n);
    assert(blas_status == CUBLAS_STATUS_SUCCESS);
    blas_status = cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_qr, n, dev_q, n, &beta, dev_matrix, n);
    assert(blas_status == CUBLAS_STATUS_SUCCESS);

    // Compute L_k = Q_k * Q_(k-1)..Q_0 --> L_k converges to eigenvectors of A_0.
    blas_status = cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_eigvecs, n, dev_q, n, &beta, dev_qr, n);
    cuda::copy_on_device(dev_eigvecs, dev_qr, matrix.size());
    cuda::copy_on_device(dev_qr, dev_matrix, matrix.size());
  }

  std::cout << "\nEigenvalue matrix:\n";
  print_device_matrix(dev_matrix, n);

  std::cout<< "\nEigenvector matrix:\n";
  print_device_matrix(dev_eigvecs, n);

  cuda::free(dev_eigvecs);

  cuda::free(dev_q);
  cuda::free(dev_info);
  cuda::free(dev_workspace);
  cuda::free(dev_tau);
  cuda::free(dev_qr);
  cuda::free(dev_matrix);

  if (blas_handle) cublasDestroy(blas_handle);
  if (solver_handle) cusolverDnDestroy(solver_handle);

  cuda::device_reset();
}
