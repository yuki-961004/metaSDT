#ifndef MODEL_BCH_HPP
#define MODEL_BCH_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Bayesian Confidence Hypothesis Model (BCH)
template <typename T>
class ModelBCH {
private:
    std::vector<T> d_vec;
    T sd_noise;
    T sd_signal;
    std::vector<T> p_thresholds; // Probability-scale thresholds
    std::vector<std::vector<T>> criteria_matrix; // Per-condition evidence-scale criteria

public:
    explicit ModelBCH(
        const std::unordered_map<std::string, std::vector<T>>& std_params
    );

    const std::vector<T>& get_p_thresholds() const { return p_thresholds; }
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
};

#endif // MODEL_BCH_HPP
