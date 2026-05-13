#include <Rcpp.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace r_modify_outputs {

inline void collect_param_order(
    SEXP obj,
    std::vector<std::string>& ordered_params,
    std::unordered_map<std::string, size_t>& param_sizes
) {
    if (Rf_isNull(obj) || Rf_length(obj) == 0) return;
    Rcpp::RObject rx(obj);
    if (!rx.hasAttribute("names")) return;
    Rcpp::CharacterVector nms = rx.attr("names");

    if (Rcpp::is<Rcpp::List>(rx)) {
        Rcpp::List lx(rx);
        for (int i = 0; i < lx.size(); ++i) {
            std::string k = Rcpp::as<std::string>(nms[i]);
            if (param_sizes.find(k) == param_sizes.end()) {
                ordered_params.push_back(k);
            }
            param_sizes[k] =
                static_cast<size_t>(Rcpp::as<std::vector<double>>(lx[i]).size());
        }
    } else if (Rcpp::is<Rcpp::NumericVector>(rx)) {
        Rcpp::NumericVector vx(rx);
        for (int i = 0; i < vx.size(); ++i) {
            std::string k = Rcpp::as<std::string>(nms[i]);
            if (param_sizes.find(k) == param_sizes.end()) {
                ordered_params.push_back(k);
            }
            param_sizes[k] = 1;
        }
    }
}

inline std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out = user_order;
    std::unordered_map<std::string, bool> seen;
    for (const auto& k : out) seen[k] = true;
    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) remain.push_back(kv.first);
    }
    std::sort(remain.begin(), remain.end());
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

inline std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;
    auto bases = ordered_base_names(user_order, best_params);
    for (const auto& b : bases) {
        size_t p_len = 0;
        auto its = param_sizes.find(b);
        if (its != param_sizes.end()) p_len = its->second;
        auto itb = best_params.find(b);
        if (itb != best_params.end()) {
            p_len = std::max(p_len, itb->second.size());
        }
        if (p_len <= 1) {
            out.push_back(b);
        } else {
            for (size_t j = 0; j < p_len; ++j) {
                out.push_back(b + "_" + std::to_string(j + 1));
            }
        }
    }
    return out;
}

} // namespace r_modify_outputs

