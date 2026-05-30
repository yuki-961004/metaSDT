#ifndef SHELL_RUN_M_HPP
#define SHELL_RUN_M_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <metaSDT/modify_params.hpp>

struct ShellRunMOptions {
    int n = 1000;
    int seed = 1004;
    int density_points = 512;
    bool has_seed = false;
    bool has_xlim = false;
    std::vector<double> xlim;
};

struct ShellRunMData {
    std::vector<int> trial;
    std::vector<int> stim;
    std::vector<int> resp;
    std::vector<int> conf;
    std::vector<int> diff;
    std::vector<double> evidence;
};

struct ShellRunMDensity {
    std::vector<double> x;
    std::vector<double> noise;
    std::vector<double> signal;
};

struct ShellRunMResult {
    std::string model;
    std::unordered_map<std::string, std::vector<double>> params;
    ShellRunMData data;
    ShellRunMDensity density;
    std::vector<double> criteria;
    int seed = 1004;
};

ShellRunMResult shell_run_m(
    const ParamGroup& params,
    const std::string& model,
    const ShellRunMOptions& option
);

#endif // SHELL_RUN_M_HPP
