#ifndef MODEL_SDT_HPP
#define MODEL_SDT_HPP

#include <metaSDT/matrix_prob.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// 标准信号检测论模型 (Standard SDT)
template <typename T>
class ModelSDT {
private:
    std::vector<T> d_vec;
    T sd_noise;
    T sd_signal;
    std::vector<T> mu_noise_vec;
    std::vector<T> mu_signal_vec;
    std::vector<T> criteria;
    std::unordered_map<std::string, std::vector<T>> std_params;

public:
    explicit ModelSDT(
        const std::unordered_map<std::string, std::vector<T>>& params
    );

    const std::vector<T>& get_criteria() const {
        return criteria;
    }

    std::vector<std::vector<T>> cdf_noise() const;
    std::vector<std::vector<T>> cdf_signal() const;

    T cdf_noise(T x, std::size_t dim_idx) const;
    T cdf_signal(T x, std::size_t dim_idx) const;

    std::vector<T> cdf_noise(
        const std::vector<T>& x_vec,
        std::size_t dim_idx
    ) const;
    std::vector<T> cdf_signal(
        const std::vector<T>& x_vec,
        std::size_t dim_idx
    ) const;

    T area(
        std::size_t stimulus,
        std::size_t response,
        const T& lower,
        const T& upper,
        std::size_t dim_idx
    ) const;

    MatrixProb<T> area() const;
};

#endif // MODEL_SDT_HPP