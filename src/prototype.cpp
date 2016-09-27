#include <cassert>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "geigen/geigen.h"

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

  std::vector<float> eigvecs(n * n);
  std::vector<float> eigvals(n * n);
  std::tie(eigvecs, eigvals) = geigen::compute_eigensystem(matrix, n);

  std::cout << "\nEigenvalue matrix:\n";
  print_matrix(eigvals, n);

  std::cout<< "\nEigenvector matrix:\n";
  print_matrix(eigvecs, n);
}
