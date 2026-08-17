#ifndef MODEL_PROBABILITIES_HPP
#define MODEL_PROBABILITIES_HPP

#include <metaSDT/matrix_prob.hpp>

#include <string>
#include <unordered_map>
#include <vector>

// Core model dispatcher for all metaSDT models
template <typename T>
MatrixProb<T> model_probabilities(
    const std::string& model_id,
    const std::unordered_map<std::string, std::vector<T>>& params
);

#endif // MODEL_PROBABILITIES_HPP
