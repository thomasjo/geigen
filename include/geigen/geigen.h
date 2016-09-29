#pragma once

#include <tuple>
#include <vector>

namespace geigen {

template<typename T>
using eigensystem = std::tuple<std::vector<T>, std::vector<T>>;

eigensystem<float> compute_eigensystem(const std::vector<float>& matrix, const int n);

}
