#ifndef MODEL_BCH_HPP
#define MODEL_BCH_HPP

#include <metaSDT/matrix_prob.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// 贝叶斯置信度假说模型 (Bayesian Confidence Hypothesis Model)
template <typename T>
class ModelBCH {
private:
    std::vector<T> d_vec;
    T sd_noise;
    T sd_signal;
    T p_resp;
    std::vector<T> p_thresholds;
    std::vector<std::vector<T>> criteria_matrix;
    std::unordered_map<std::string, std::vector<T>> std_params;

public:
    explicit ModelBCH(
        const std::unordered_map<std::string, std::vector<T>>& params
    );

    const std::vector<T>& get_p_thresholds() const {
        return p_thresholds;
    }

    const std::vector<std::vector<T>>& get_criteria_matrix() const {
        return criteria_matrix;
    }

    std::vector<std::vector<T>> cdf_noise() const;
    std::vector<std::vector<T>> cdf_signal() const;

    T area(
        std::size_t stimulus,
        std::size_t response,
        const T& lower,
        const T& upper,
        std::size_t dim_idx
    ) const;

    std::vector<std::vector<std::vector<T>>> compute_probabilities() const;

    MatrixProb<T> area() const;
};

#endif // MODEL_BCH_HPP
