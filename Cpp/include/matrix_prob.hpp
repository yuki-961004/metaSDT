#ifndef MATRIX_PROB_HPP
#define MATRIX_PROB_HPP

#include <vector>
#include <string>
#include <unordered_map>

template <typename T>
struct MatrixProb {
    std::vector<std::vector<std::vector<T>>> prob_mat; // 升级为 3D
    std::vector<std::string> dim_names;
    std::vector<std::string> row_names;
    std::vector<std::string> col_names;
};

template <typename T>
MatrixProb<T> matrix_prob(
    const std::vector<std::vector<T>>& cdf_noise,
    const std::vector<std::vector<T>>& cdf_signal,
    const std::unordered_map<std::string, std::vector<T>>& std_params
);

#endif // MATRIX_PROB_HPP