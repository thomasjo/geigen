#include "geigen/geigen.h"

#include <cassert>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cudalicious/blas.hpp>
#include <cudalicious/core.hpp>
#include <cudalicious/solver.hpp>

#include <magma_v2.h>

constexpr auto MAX_ITER = 50;

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
void construct_q_matrix(float* temp, const float* q, const float* source, const float* tau, const int n, const int k)
{
  // Remember that `x` gives the row index, and `y` gives the column index.
  const auto row = blockDim.x * blockIdx.x + threadIdx.x;
  const auto col = blockDim.y * blockIdx.y + threadIdx.y;
  const auto idx = col * n + row;

  if (row >= n || col >= n) return;

  // for (auto k = 0; k < n; ++k) {
    auto inner_product = 0.0f;

    for (auto i = 0; i < n; ++i) {
      const auto row_coefficient = q[i * n + row];
      const auto col_coefficient = get_reflector_coefficient(source, tau, n, k, i, col);

      inner_product += row_coefficient * col_coefficient;
    }

    temp[idx] = inner_product;
  // }
}

geigen::eigensystem<float> geigen::compute_eigensystem(const std::vector<float>& matrix, const int n)
{
  assert(matrix.size() == static_cast<size_t>(n * n));

  size_t free_mem;
  size_t total_mem;
  cuda::check_error(cudaMemGetInfo(&free_mem, &total_mem));

  std::cout << " Free memory: " << free_mem << "\n";
  std::cout << "Total memory: " << total_mem << "\n";

  // Initialize MAGMA.
  magma_init();
  magma_int_t info = 0;

  // Determine block size.
  // auto nb = magma_get_sgeqrf_nb(n, n);
  auto nb = magma_get_chetrd_nb(n);
  std::cout << "nb: " << nb << "\n";

  std::vector<magmaFloatComplex> cmatrix;
  for (auto num : matrix) {
    // std::cout << MAGMA_C_MAKE(num, 0).x << "\n";
    cmatrix.push_back(MAGMA_C_MAKE(num, 0));
  }

  // std::vector<magmaFloatComplex> cmatrix(n * n);
  // for (auto col = 0; col < n; ++col) {
  //   for (auto row = 0; row < n; ++row) {
  //     if (col < row) continue;
  //     const auto idx = row * n + col;
  //     cmatrix[idx] = MAGMA_C_MAKE(matrix[idx], 0);
  //   }
  // }

  std::cout << "size: " << cmatrix.size() << "\n";
  std::cout << cmatrix[0].x << "\n";

  std::vector<float> eigvals(n);
  // std::vector<magmaFloatComplex> work_matrix(n * n);
  // std::vector<magmaFloatComplex> work_eigvals((nb + 1) * n);
  // std::vector<float> work_real(1 + 5*n + 2*n*n);
  // std::vector<int> work_int(3 + 5*n);
  // std::vector<int> fail(n);

  int num_eigvals;

  // Allocate device memory.
  magmaFloatComplex_ptr dev_matrix;
  magma_cmalloc(&dev_matrix, n * n);
  magmaFloatComplex_ptr dev_eigvecs;
  magma_cmalloc(&dev_eigvecs, n * n);

  // Copy from CPU to GPU...
  magma_device_t device;
  magma_getdevice(&device);
  magma_queue_t queue;
  magma_queue_create(device, &queue);
  magma_csetmatrix(n, n, cmatrix.data(), n, dev_matrix, n, queue);

  magma_sprint(    4, 4, matrix.data(),  n);
  magma_cprint(    4, 4, cmatrix.data(), n);
  magma_cprint_gpu(4, 4, dev_matrix,     n);

  // magmaFloatComplex lwork;
  // float lrwork;
  // int liwork;

  // magma_cheevdx(
  //   MagmaVec,  // jobs [in]
  //   MagmaRangeAll,  // range [in]
  //   MagmaLower,  // uplo [in]
  //   n,  // n [in]
  //   cmatrix.data(),  // dA [in,out]
  //   n,  // ldda [in]
  //   0,  // vl [in]
  //   10,  // vu [in]
  //   0,  // il [in]
  //   n - 1,  // iu [in]
  //   &num_eigvals,  // m [out]
  //   eigvals.data(),  // w [out]
  //   &lwork,  // work [out]
  //   -1,  // lwork [in]
  //   &lrwork,  // rwork [out]
  //   -1, // lrwork [in]
  //   &liwork,  // iwork [out]
  //   -1,  // liwork [out]
  //   &info  // info [out]
  // );

  // std::cout << "lwork: " << lwork.x << "\n";
  // std::cout << "lrwork: " << lrwork << "\n";
  // std::cout << "liwork: " << liwork << "\n";

  std::vector<magmaFloatComplex> work_matrix(n * n);
  std::vector<magmaFloatComplex> work_eigvecs(n * n);
  // std::vector<magmaFloatComplex> work(2*n - 1);
  std::vector<float> work_real(7*n);
  std::vector<int> work_int(5*n);
  std::vector<int> fail(n);

  // std::cout << "work_eigvals.size(): " << work_eigvals.size() << "\n";

  magmaFloatComplex lwork;

  // magma_cheevx(
  //   MagmaNoVec,           // jobz [in]
  //   MagmaRangeAll,        // range [in]
  //   MagmaLower,           // uplo [in]
  //   n,                    // n [in]
  //   cmatrix.data(),       // A [in,out]
  //   n,                    // lda [in]
  //   0,                    // vl [in]
  //   0,                    // vu [in]
  //   0,                    // il [in]
  //   0,                    // iu [in]
  //   0.0000001,            // abstol [in]
  //   &num_eigvals,         // m [out]
  //   eigvals.data(),       // w [out]
  //   work_eigvecs.data(),  // Z [out]
  //   n,                    // ldz [in]
  //   &lwork,               // work [out]
  //   -1,                   // lwork [in]
  //   work_real.data(),     // rwork [out]
  //   work_int.data(),      // iwork [out]
  //   fail.data(),          // ifail [out]
  //   &info                 // info [out]
  // );

  magma_cheevx_gpu(
    MagmaNoVec,           // jobz [in]
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

  // std::cout << work_eigvals[0].x << "\n";

  std::vector<magmaFloatComplex> work(lwork.x);

  // magma_cheevx(
  //   MagmaNoVec,           // jobz [in]
  //   MagmaRangeAll,        // range [in]
  //   MagmaLower,           // uplo [in]
  //   n,                    // n [in]
  //   cmatrix.data(),       // A [in,out]
  //   n,                    // lda [in]
  //   0,                    // vl [in]
  //   0,                    // vu [in]
  //   0,                    // il [in]
  //   0,                    // iu [in]
  //   0.0000001,            // abstol [in]
  //   &num_eigvals,         // m [out]
  //   eigvals.data(),       // w [out]
  //   work_eigvecs.data(),  // Z [out]
  //   n,                    // ldz [in]
  //   work.data(),          // work [out]
  //   work.size(),          // lwork [in]
  //   work_real.data(),     // rwork [out]
  //   work_int.data(),      // iwork [out]
  //   fail.data(),          // ifail [out]
  //   &info                 // info [out]
  // );

  magma_cheevx_gpu(
    MagmaNoVec,           // jobz [in]
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

  std::cout << "num_eigvals: " << num_eigvals << "\n";

  std::vector<float> eigvecs(n * n, 0);

  // std::cout << "failed indices: \n";
  // for (const auto idx : fail) {
  //   std::cout << idx << ", ";
  // }
  // std::cout << "\n";

  magma_queue_destroy(queue);

  // Release device memory.
  magma_free(dev_eigvecs);
  magma_free(dev_matrix);

  magma_finalize();
  cuda::device_reset();

  return std::make_tuple(eigvals, eigvecs);
}
