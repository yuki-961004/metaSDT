#include "abcpp/options.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace abcpp {

namespace {

std::string lower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        }
    );
    return value;
}

}  // namespace

Method parse_method(const std::string& value) {
    const std::string key = lower(value);
    if (key == "rejection") {
        return Method::Rejection;
    }
    if (key == "loclinear") {
        return Method::LocLinear;
    }
    if (key == "ridge") {
        return Method::Ridge;
    }
    if (key == "neuralnet") {
        return Method::NeuralNet;
    }
    throw std::invalid_argument("Unknown ABC method.");
}

Kernel parse_kernel(const std::string& value) {
    const std::string key = lower(value);
    if (key == "gaussian") {
        return Kernel::Gaussian;
    }
    if (key == "epanechnikov") {
        return Kernel::Epanechnikov;
    }
    if (key == "rectangular") {
        return Kernel::Rectangular;
    }
    if (key == "triangular") {
        return Kernel::Triangular;
    }
    if (key == "biweight") {
        return Kernel::Biweight;
    }
    if (key == "cosine") {
        return Kernel::Cosine;
    }
    throw std::invalid_argument("Unknown ABC kernel.");
}

Transform parse_transform(const std::string& value) {
    const std::string key = lower(value);
    if (key == "none") {
        return Transform::None;
    }
    if (key == "log") {
        return Transform::Log;
    }
    if (key == "logit") {
        return Transform::Logit;
    }
    throw std::invalid_argument("Unknown parameter transformation.");
}

ReductionMethod parse_reduction(const std::string& value) {
    const std::string key = lower(value);
    if (key.empty() || key == "none" || key == "null") {
        return ReductionMethod::None;
    }
    if (key == "pca") {
        return ReductionMethod::PCA;
    }
    if (key == "pls") {
        return ReductionMethod::PLS;
    }
    throw std::invalid_argument("Unknown summary reduction method.");
}

std::string method_name(Method method) {
    switch (method) {
    case Method::Rejection:
        return "rejection";
    case Method::LocLinear:
        return "loclinear";
    case Method::Ridge:
        return "ridge";
    case Method::NeuralNet:
        return "neuralnet";
    }
    return "unknown";
}

std::string kernel_name(Kernel kernel) {
    switch (kernel) {
    case Kernel::Gaussian:
        return "gaussian";
    case Kernel::Epanechnikov:
        return "epanechnikov";
    case Kernel::Rectangular:
        return "rectangular";
    case Kernel::Triangular:
        return "triangular";
    case Kernel::Biweight:
        return "biweight";
    case Kernel::Cosine:
        return "cosine";
    }
    return "unknown";
}

std::string transform_name(Transform transform) {
    switch (transform) {
    case Transform::None:
        return "none";
    case Transform::Log:
        return "log";
    case Transform::Logit:
        return "logit";
    }
    return "unknown";
}

std::string reduction_name(ReductionMethod reduction) {
    switch (reduction) {
    case ReductionMethod::None:
        return "none";
    case ReductionMethod::PCA:
        return "PCA";
    case ReductionMethod::PLS:
        return "PLS";
    }
    return "unknown";
}

}  // namespace abcpp
