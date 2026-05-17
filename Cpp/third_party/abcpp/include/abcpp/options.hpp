#pragma once

#include "abcpp/matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace abcpp {

enum class Method {
    Rejection,
    LocLinear,
    Ridge,
    NeuralNet
};

using AbcMethod = Method;

enum class Kernel {
    Gaussian,
    Epanechnikov,
    Rectangular,
    Triangular,
    Biweight,
    Cosine
};

enum class Transform {
    None,
    Log,
    Logit
};

enum class ReductionMethod {
    None,
    PCA,
    PLS
};

struct ReductionOptions {
    ReductionMethod method = ReductionMethod::None;
    std::size_t ncomp = 0;
};

struct AbcOptions {
    double tol = 0.1;
    Method method = Method::Rejection;
    bool hcorr = true;
    std::vector<Transform> transformations;
    Matrix logit_bounds;
    std::vector<bool> subset;
    Kernel kernel = Kernel::Epanechnikov;
    int numnet = 10;
    int sizenet = 5;
    std::vector<double> lambda = {0.0001, 0.001, 0.01};
    int maxit = 500;
    unsigned int seed = 1004;
    ReductionOptions reduction;
};

Method parse_method(const std::string& value);

Kernel parse_kernel(const std::string& value);

Transform parse_transform(const std::string& value);

ReductionMethod parse_reduction(const std::string& value);

std::string method_name(Method method);

std::string kernel_name(Kernel kernel);

std::string transform_name(Transform transform);

std::string reduction_name(ReductionMethod reduction);

}  // namespace abcpp
