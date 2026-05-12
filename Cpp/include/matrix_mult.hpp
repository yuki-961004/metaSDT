#ifndef MATRIX_MULT_HPP
#define MATRIX_MULT_HPP

#include <vector>
#include <string>
#include <unordered_map>

template <typename T>
std::vector<std::vector<std::vector<T>>> matrix_mult(
    const std::vector<std::vector<std::vector<double>>>& freq_mat,
    const std::vector<std::vector<std::vector<T>>>& prob_mat,
    const std::unordered_map<std::string, std::vector<T>>& std_params
);

#endif // MATRIX_MULT_HPP