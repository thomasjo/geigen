#include <cassert>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "geigen/geigen.h"

constexpr auto PRECISION = 7U;
constexpr auto COLUMN_WIDTH = 12U;

void print_header(const std::string& text, const size_t width)
{
  const auto text_border = "\n" + std::string(width, '=') + "\n";
  std::cout << text_border << text << text_border;
}

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
void print_eigensystem(const geigen::eigensystem<T>& eigensystem, const size_t n)
{
  std::cout << "\nEigenvalues:\n";
  for (const auto val : eigensystem.values) { std::cout << val << "\n"; }

  std::cout<< "\nEigenvector matrix:\n";
  print_matrix(eigensystem.vectors, n);
}


template<typename T>
void compute_eigensystem(const std::vector<T>& matrix, const int n)
{
  print_header("Computing eigensystem...", COLUMN_WIDTH * n);

  const auto eigensystem = geigen::compute_eigensystem(matrix, n);
  print_eigensystem(eigensystem, n);
}

template<typename T>
void compute_eigensystem_magma(const std::vector<T>& matrix, const int n)
{
  print_header("Computing eigensystem using MAGMA...", COLUMN_WIDTH * n);

  const auto eigensystem = geigen::compute_eigensystem_magma(matrix, n);
  print_eigensystem(eigensystem, n);
}

int main()
{
  constexpr auto n = 4U;
  const std::vector<float> matrix {
     5, -2, -1, 0,
    -2,  5,  0, 1,
    -1,  0,  5, 2,
     0,  1,  2, 5,
  };
  assert(matrix.size() == n * n);

  std::cout << "Input matrix:\n";
  print_matrix(matrix, n);

  compute_eigensystem(matrix, n);
  compute_eigensystem_magma(matrix, n);
}
