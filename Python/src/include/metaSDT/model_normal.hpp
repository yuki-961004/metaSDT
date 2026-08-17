#ifndef MODEL_NORMAL_HPP
#define MODEL_NORMAL_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Gaussian Metacognitive Noise Model (Normal)
template <typename T>
class ModelNormal {
private:
    std::vector<T> d_vec;
    T c_resp;
    std::vector<T> c_conf;
    T sigma_meta;
    T sd_noise;
    T sd_signal;
    std::vector<T> criteria; // Canonical full criteria vector

public:
    explicit ModelNormal(
        const std::unordered_map<std::string, std::vector<T>>& std_params
    );

    const std::vector<T>& get_criteria() const { return criteria; }

    std::vector<std::vector<std::vector<T>>> compute_probabilities() const;

    T area(
        std::size_t stimulus,
        std::size_t response,
        const T& lower,
        const T& upper,
        std::size_t dim_idx
    ) const;
};

#endif // MODEL_NORMAL_HPP
