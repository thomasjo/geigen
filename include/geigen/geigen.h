#pragma once

#include <tuple>
#include <vector>

namespace geigen {

template<typename T>
struct eigensystem
{
  std::vector<T> values;
  std::vector<T> vectors;

  eigensystem(std::vector<T> values, std::vector<T> vectors) : values(values), vectors(vectors) {}
};

eigensystem<float> compute_eigensystem(const std::vector<float>& matrix, const int n);
eigensystem<float> compute_eigensystem_magma(const std::vector<float>& matrix, const int n);

}
