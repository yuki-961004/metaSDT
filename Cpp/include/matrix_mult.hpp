#ifndef MATRIX_MULT_HPP
#define MATRIX_MULT_HPP

#include <vector>
#include <string>
#include <unordered_map>

// 主函数声明
std::vector<std::vector<double>> matrix_mult(
    const std::vector<std::vector<double>>& freq_mat,
    const std::vector<std::vector<double>>& prob_mat,
    const std::unordered_map<std::string, std::vector<double>>& params
);

#endif // MATRIX_MULT_HPP