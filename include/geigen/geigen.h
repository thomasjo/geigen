#pragma once

#include <tuple>
#include <vector>

namespace geigen {

std::tuple<std::vector<float>, std::vector<float>>
compute_eigensystem(const std::vector<float>& matrix, const size_t n);

}
