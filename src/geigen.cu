#include "geigen/geigen.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cudalicious/blas.hpp>
#include <cudalicious/core.hpp>
#include <cudalicious/solver.hpp>

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
  std::vector<T> v(n * n);
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

  // if (idx >= n * n) return;
  if (row >= n || col >= n) return;

  // printf("(% u, % u) -- % u\n", row, col, idx);

  for (auto k = 0; k < n; ++k) {
    auto inner_product = 0.0f;

    for (auto i = 0; i < n; ++i) {
      const auto row_coefficient = q[i * n + row];
      const auto col_coefficient = get_reflector_coefficient(source, tau, n, k, i, col);
      // const auto col_coefficient = 0.f;

      inner_product += row_coefficient * col_coefficient;
    }

    q[idx] = inner_product;
  }
}

std::tuple<std::vector<float>, std::vector<float>>
geigen::compute_eigensystem(const std::vector<float>& matrix, const int n)
{
  size_t free_mem;
  size_t total_mem;
  cuda::check_error(cudaMemGetInfo(&free_mem, &total_mem));

  std::cout << " Free memory: " << free_mem << "\n";
  std::cout << "Total memory: " << total_mem << "\n";

  cuda::device_sync();

  assert(matrix.size() == static_cast<size_t>(n * n));

  std::vector<float> identity(n * n, 0);
  for (auto i = 0; i < n; ++i) {
    identity[i * n + i] = 1.f;
  }
  assert(identity.size() == matrix.size());

  // std::cout << "-- " << identity[0] << "\n";
  // std::cout << "-- " << identity[1 * n + 1] << "\n";
  // std::cout << "-- " << identity[2 * n + 2] << "\n";
  // std::cout << "-- " << identity[2 * n + 3] << "\n";

  // Create handles.
  // std::cout << "Creating handles...\n";
  auto solver_handle = cuda::solver::initialize();
  auto blas_handle = cuda::blas::initialize();

  // Allocate device memory.
  // std::cout << "Copying matrix to device...\n";
  auto dev_matrix = cuda::copy_to_device(matrix);

  // Determine workspace size.
  // std::cout << "Determining workspace size...\n";
  auto workspace_size = cuda::solver::geqrf_buffer_size(solver_handle, n, n, dev_matrix, n);
  // std::cout << "Workspace size: " << workspace_size << "\n";

  // std::cout << "Preparing device memory...\n";
  auto dev_qr = cuda::allocate<float>(matrix.size());
  auto dev_tau = cuda::allocate<float>(n);
  auto dev_workspace = cuda::allocate<float>(workspace_size);
  auto dev_info = cuda::allocate<int>(1);

  // auto dev_q = cuda::copy_to_device(identity);
  auto dev_q = cuda::allocate<float>(matrix.size());
  auto dev_eigvecs = cuda::copy_on_device(dev_q, matrix.size());

  // std::cout << "Copying matrix on device...\n";
  cuda::copy_on_device(dev_qr, dev_matrix, matrix.size());

  for (auto k = 0; k < K_MAX; ++k) {
    // Compute QR factorization.
    // std::cout << "Computing QR factorization...\n";
    cuda::solver::geqrf(solver_handle, n, n, dev_qr, n, dev_tau, dev_workspace, workspace_size, dev_info);

    // std::cout << "Copying identity matrix to device...\n" << std::flush;
    cuda::copy_to_device(dev_q, identity.data(), identity.size());

    // std::cout << "Launching kernel...\n" << std::flush;

    const dim3 threads_per_block(16, 16);
    // const dim3 threads_per_block(32, 32);
    const dim3 blocks(
      std::ceil(n / threads_per_block.x) + (((n % threads_per_block.x) == 0 ? 0 : 1)),
      std::ceil(n / threads_per_block.y) + (((n % threads_per_block.y) == 0 ? 0 : 1))
    );

    // std::cout << n * n << "\n";
    // const int blocks(10);
    // const int threads_per_block(1);
    // std::cout << "Hmm...\n" << std::flush;
    // cuda::device_sync();

    construct_q_matrix<<<blocks, threads_per_block>>>(dev_q, dev_qr, dev_tau, n);
    cuda::device_sync();

    constexpr auto alpha = 1.0f;
    constexpr auto beta = 0.0f;

    // Compute A_k = Q_k^T * A_(k-1) * Q_k --> A_k converges to eigenvalues of A_0.
    cuda::blas::gemm(blas_handle, n, n, n, alpha, dev_q, n, dev_matrix, n, beta, dev_qr, n, true);
    cuda::blas::gemm(blas_handle, n, n, n, alpha, dev_qr, n, dev_q, n, beta, dev_matrix, n);

    // Compute L_k = Q_k * Q_(k-1)..Q_0 --> L_k converges to eigenvectors of A_0.
    cuda::blas::gemm(blas_handle, n, n, n, alpha, dev_eigvecs, n, dev_q, n, beta, dev_qr, n);

    cuda::copy_on_device(dev_eigvecs, dev_qr, matrix.size());
    cuda::copy_on_device(dev_qr, dev_matrix, matrix.size());
  }

  std::vector<float> eigvecs(n * n);
  cuda::copy_to_host(eigvecs, dev_eigvecs);

  std::vector<float> eigvals(n * n);
  cuda::copy_to_host(eigvals, dev_matrix);

  cuda::free(dev_eigvecs);
  cuda::free(dev_q);
  cuda::free(dev_info);
  cuda::free(dev_workspace);
  cuda::free(dev_tau);
  cuda::free(dev_qr);
  cuda::free(dev_matrix);

  cuda::blas::release(blas_handle);
  cuda::solver::release(solver_handle);

  cuda::device_reset();

  return std::make_tuple(eigvals, eigvecs);
}
