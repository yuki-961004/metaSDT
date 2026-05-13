#include <Rcpp.h>
#include "../../Cpp/include/data_info.hpp"

#define CORE_IMPL "../../Cpp/src/data_info.cpp"
#include CORE_IMPL
//' Intelligently scan the dataset and extract subject-level information.
// [[Rcpp::export(name = "data_info")]]
Rcpp::List r_data_info(
    Rcpp::DataFrame df, 
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue
) {
    
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    Rcpp::CharacterVector df_names = df.names();
    
    for (int i = 0; i < df.size(); ++i) {
        int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            cpp_df[Rcpp::as<std::string>(df_names[i])] = 
                Rcpp::as<std::vector<double>>(df[i]);
        }
    }
    
    std::unordered_map<std::string, std::string> cpp_colnames;
    if (colnames.isNotNull()) {
        Rcpp::List col_list(colnames);
        if (col_list.hasAttribute("names")) {
            Rcpp::CharacterVector c_names = col_list.names();
            for (int i = 0; i < col_list.size(); ++i) {
                cpp_colnames[Rcpp::as<std::string>(c_names[i])] = 
                    Rcpp::as<std::string>(col_list[i]);
            }
        }
    }
    
    DataInfoResult res;
    try {
        res = ::data_info(/*df=*/cpp_df, /*colnames=*/cpp_colnames);
    } catch (std::exception& e) {
        Rcpp::stop(e.what());
    }
    
    Rcpp::Function r_message("message");
    for (const auto& msg : res.messages) {
        r_message(msg);
    }
    
    Rcpp::List r_colnames;
    for (const auto& kv : res.colnames) {
        r_colnames[kv.first] = kv.second;
    }
    
    Rcpp::List r_subjects;
    for (const auto& kv : res.subjects) {
        double subid = kv.first;
        
        std::string sub_key = std::to_string(subid);
        sub_key.erase(sub_key.find_last_not_of('0') + 1, std::string::npos);
        if (!sub_key.empty() && sub_key.back() == '.') {
            sub_key.pop_back();
        }
        
        const DataInfoSubject& subj = kv.second;
        
        Rcpp::IntegerVector r_raw(subj.raw.size());
        for (size_t i = 0; i < subj.raw.size(); ++i) {
            r_raw[i] = subj.raw[i] + 1; 
        }
        
        Rcpp::List r_condition;
        for (const auto& gkv : subj.condition) {
            Rcpp::IntegerVector g_idx(gkv.second.size());
            for (size_t i = 0; i < gkv.second.size(); ++i) {
                g_idx[i] = gkv.second[i] + 1;
            }
            r_condition[gkv.first] = g_idx;
        }
        
        Rcpp::List r_difficulty;
        for (const auto& dkv : subj.difficulty) {
            Rcpp::IntegerVector d_idx(dkv.second.size());
            for (size_t i = 0; i < dkv.second.size(); ++i) {
                d_idx[i] = dkv.second[i] + 1;
            }
            r_difficulty[dkv.first] = d_idx;
        }
        
        Rcpp::List r_info = Rcpp::List::create(
            Rcpp::Named("n_trials") = subj.info.n_trials,
            Rcpp::Named("n_blocks") = subj.info.n_blocks
        );
        
        r_subjects[sub_key] = Rcpp::List::create(
            Rcpp::Named("raw") = r_raw,
            Rcpp::Named("condition") = r_condition,
            Rcpp::Named("difficulty") = r_difficulty,
            Rcpp::Named("info") = r_info
        );
    }
    
    return Rcpp::List::create(
        Rcpp::Named("colnames") = r_colnames,
        Rcpp::Named("subjects") = r_subjects
    );
}
