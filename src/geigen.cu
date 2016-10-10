#include "geigen/geigen.h"

#include <cassert>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cudalicious/blas.hpp>
#include <cudalicious/core.hpp>
#include <cudalicious/solver.hpp>

#include <magma_v2.h>

constexpr auto PRECISION = 7U;
constexpr auto COLUMN_WIDTH = 12U;
constexpr auto MAX_ITER = 50;

template<typename T>
void print_matrix(const std::vector<T>& matrix, const int n)
{
  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      const auto idx = j * n + i;
      std::cout << std::setprecision(PRECISION) << std::setw(COLUMN_WIDTH) << matrix[idx];
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

void print_available_gpu_memory()
{
  size_t free_mem;
  size_t total_mem;
  cuda::check_error(cudaMemGetInfo(&free_mem, &total_mem));

  std::cout << " Free GPU memory: " << std::setw(12) << free_mem << "\n";
  std::cout << "Total GPU memory: " << std::setw(12) << total_mem << "\n";
}

__device__
float get_reflector_coefficient(const float* source, const float* tau, const int n, const int reflector_index, const int row, const int col)
{
  if (row < reflector_index || col < reflector_index) return (row == col) ? 1.f : 0.f;

  const auto row_coefficient = (row == reflector_index) ? 1.f : source[reflector_index * n + row];
  const auto col_coefficient = (col == reflector_index) ? 1.f : source[reflector_index * n + col];

  return ((row == col) ? 1.f : 0.f) - tau[reflector_index] * row_coefficient * col_coefficient;
}

__global__
void construct_q_matrix(float* q, const float* source, const float* tau, const size_t n)
{
  // Remember that `x` gives the row index, and `y` gives the column index.
  const auto row = blockDim.x * blockIdx.x + threadIdx.x;
  const auto col = blockDim.y * blockIdx.y + threadIdx.y;
  const auto idx = col * n + row;

  if (row >= n || col >= n) return;

  for (auto k = 0U; k < n; ++k) {
    auto inner_product = 0.f;

    for (auto i = 0U; i < n; ++i) {
      const auto row_coefficient = q[i * n + row];
      const auto col_coefficient = get_reflector_coefficient(source, tau, n, k, i, col);

      inner_product += row_coefficient * col_coefficient;
    }

    __syncthreads();
    q[idx] = inner_product;
  }
}

geigen::eigensystem<float> geigen::compute_eigensystem(const std::vector<float>& matrix, const size_t n)
{
  assert(matrix.size() == n * n);

  // Because of synchronization issues, the maximum matrix size is 32x32, since the limit is 1024 threads per block,
  // and we only support a single block since we need `__syncthreads()`.
  if (n > 32U) {
    std::cerr << "Maximum supported matrix size is 32x32!\n";
    std::exit(1);
  }

#ifndef NDEBUG
  print_available_gpu_memory();
#endif

  const dim3 threads_per_block(32, 32);
  const dim3 block_size(
    std::ceil(n / threads_per_block.x) + (((n % threads_per_block.x) == 0 ? 0 : 1)),
    std::ceil(n / threads_per_block.y) + (((n % threads_per_block.y) == 0 ? 0 : 1))
  );

  // Create handles.
  auto solver_handle = cuda::solver::initialize();
  auto blas_handle = cuda::blas::initialize();

  // Allocate device memory.
  auto dev_matrix = cuda::copy_to_device(matrix);
  auto dev_qr = cuda::allocate<float>(matrix.size());
  cuda::copy_on_device(dev_qr, dev_matrix, matrix.size());

  // Determine workspace size.
  auto workspace_size = cuda::solver::geqrf_buffer_size(solver_handle, n, n, dev_matrix, n);

  auto dev_tau = cuda::allocate<float>(n);
  auto dev_workspace = cuda::allocate<float>(workspace_size);
  auto dev_info = cuda::allocate<int>(1);

  std::vector<float> identity(n * n, 0);
  for (auto i = 0U; i < n; ++i) {
    identity[i * n + i] = 1.f;
  }

  auto dev_q = cuda::copy_to_device(identity);
  auto dev_eigvecs = cuda::copy_on_device(dev_q, matrix.size());

  for (auto iter = 0; iter < MAX_ITER; ++iter) {
    // Compute QR factorization.
    cuda::solver::geqrf(solver_handle, n, n, dev_qr, n, dev_tau, dev_workspace, workspace_size, dev_info);

    cuda::copy_to_device(dev_q, identity.data(), identity.size());
    construct_q_matrix<<<block_size, threads_per_block>>>(dev_q, dev_qr, dev_tau, n);
    cuda::device_sync();

    constexpr auto alpha = 1.f;
    constexpr auto beta = 0.f;

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

  auto trace = [](const std::vector<float>& matrix, const size_t n) {
    std::vector<float> tr;
    for (auto i = 0U; i < n; ++i) { tr.emplace_back(matrix[i * n + i]); }
    return tr;
  };

  return geigen::eigensystem<float>(trace(eigvals, n), eigvecs);
}

geigen::eigensystem<float> geigen::compute_eigensystem_magma(const std::vector<float>& matrix, const size_t n)
{
  assert(matrix.size() == n * n);

#ifndef DEBUG
  print_available_gpu_memory();
#endif

  // Initialize MAGMA.
  magma_init();
  magma_int_t info = 0;

  std::vector<magmaFloatComplex> cmatrix;
  for (auto num : matrix) {
    cmatrix.push_back(MAGMA_C_MAKE(num, 0));
  }

  // Allocate device memory.
  magmaFloatComplex_ptr dev_matrix;
  magma_cmalloc(&dev_matrix, n * n);

  // Copy from CPU to GPU...
  magma_device_t device;
  magma_getdevice(&device);
  magma_queue_t queue;
  magma_queue_create(device, &queue);
  magma_csetmatrix(n, n, cmatrix.data(), n, dev_matrix, n, queue);

  magma_queue_destroy(queue);

  std::vector<float> eigvals(n);
  int num_eigvals;

  std::vector<magmaFloatComplex> work_matrix(n * n);
  std::vector<magmaFloatComplex> work_eigvecs(n * n);
  std::vector<float> work_real(7 * n);  // "Magic" dimension found in MAGMA docs.
  std::vector<int> work_int(5 * n);  // "Magic" dimension found in MAGMA docs.
  std::vector<int> fail(n);

  magmaFloatComplex_ptr dev_eigvecs;
  magma_cmalloc(&dev_eigvecs, n * n);

  // Figure out what the optimal workspace size is.
  magmaFloatComplex lwork;
  magma_cheevx_gpu(
    MagmaVec,             // jobz [in]
    MagmaRangeAll,        // range [in]
    MagmaLower,           // uplo [in]
    n,                    // n [in]
    dev_matrix,           // dA [in,out]
    n,                    // ldda [in]
    0,                    // vl [in]
    0,                    // vu [in]
    0,                    // il [in]
    0,                    // iu [in]
    10e-10,               // abstol [in]
    &num_eigvals,         // m [out]
    eigvals.data(),       // w [out]
    dev_eigvecs,          // dZ [out]
    n,                    // lddz [in]
    work_matrix.data(),   // wA [in]
    n,                    // ldwa [in]
    work_eigvecs.data(),  // wZ [in]
    n,                    // ldwz [in]
    &lwork,               // work [out]
    -1,                   // lwork [in]
    work_real.data(),     // rwork [in]
    work_int.data(),      // iwork [in]
    fail.data(),          // ifail [out]
    &info                 // info [out]
  );

  // Initialize workspace with the optimal size.
  std::vector<magmaFloatComplex> work(lwork.x);

  // Execute the eigendecomposition.
  magma_cheevx_gpu(
    MagmaVec,             // jobz [in]
    MagmaRangeAll,        // range [in]
    MagmaLower,           // uplo [in]
    n,                    // n [in]
    dev_matrix,           // dA [in,out]
    n,                    // ldda [in]
    0,                    // vl [in]
    0,                    // vu [in]
    0,                    // il [in]
    0,                    // iu [in]
    10e-10,               // abstol [in]
    &num_eigvals,         // m [out]
    eigvals.data(),       // w [out]
    dev_eigvecs,          // dZ [out]
    n,                    // lddz [in]
    work_matrix.data(),   // wA [in]
    n,                    // ldwa [in]
    work_eigvecs.data(),  // wZ [in]
    n,                    // ldwz [in]
    work.data(),          // work [out]
    work.size(),          // lwork [in]
    work_real.data(),     // rwork [in]
    work_int.data(),      // iwork [in]
    fail.data(),          // ifail [out]
    &info                 // info [out]
  );

  std::vector<magmaFloatComplex> temp_eigvecs(n * n);
  cuda::copy_to_host(temp_eigvecs, dev_eigvecs);

  std::vector<float> eigvecs;
  for (auto v : temp_eigvecs) {
    eigvecs.emplace_back(MAGMA_C_REAL(v));
  }

  // Release device memory.
  magma_free(dev_eigvecs);
  magma_free(dev_matrix);

  magma_finalize();
  cuda::device_reset();

  return geigen::eigensystem<float>(eigvals, eigvecs);
}
