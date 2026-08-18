#ifndef MODEL_LOGNORMAL_HPP
#define MODEL_LOGNORMAL_HPP

#include <metaSDT/matrix_prob.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// 对数正态元认知噪声模型 (Log-normal Metacognitive Noise Model)
template <typename T>
class ModelLognormal {
private:
    std::vector<T> d_vec;
    T c_resp;
    std::vector<T> c_conf;
    T sigma_meta;
    T sd_noise;
    T sd_signal;
    std::vector<T> criteria;
    std::unordered_map<std::string, std::vector<T>> std_params;

public:
    explicit ModelLognormal(
        const std::unordered_map<std::string, std::vector<T>>& params
    );

    const std::vector<T>& get_criteria() const {
        return criteria;
    }

    std::vector<std::vector<std::vector<T>>> compute_probabilities() const;

    T area(
        std::size_t stimulus,
        std::size_t response,
        const T& lower,
        const T& upper,
        std::size_t dim_idx
    ) const;

    MatrixProb<T> area() const;
};

#endif // MODEL_LOGNORMAL_HPP
