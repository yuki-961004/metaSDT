
# %%
# 1. 编译并加载整个 R 包 (指定包根目录为 ./R)
devtools::clean_dll("./R")
devtools::load_all("./R")

# %%
data <- read.csv("data/exp1.csv")

sub1 <- data |> dplyr::filter(subj_id == 1)

#Rcpp::sourceCpp("./R/src/wrapper_data_info.cpp")

info <- info_data(data)

# 1. 从 data_info 的结果中，提取出被试 '29' 的行号索引
sub29 <- info$subjects$`29`$raw

# 2. 使用这串行号，从原始的、完整的 `data` 数据框中进行切片，得到真实的 dataframe
df_sub29 <- data[sub29, ]

# (可选) 打印一下，看看是不是成功提取了 29 号被试的数据
head(df_sub29)

# %%
#Rcpp::sourceCpp("./R/src/wrapper_matrix_freq.cpp")

freq_mat <- matrix_freq(
  stim = sub1$stim,
  resp = sub1$resp,
  conf = sub1$conf 
)

# %%
# 1. 直接编译并加载新封装的 C++ 包装器函数 (兼容从项目根目录执行)
# ==============================================================================
# 2. 测试全新的 modify_params 功能
# ==============================================================================
# 编译并加载包装器函数 (兼容从项目根目录执行)
# 注意：如果您之后使用 devtools::load_all() 整体加载了包，这行就可以删掉了
#Rcpp::sourceCpp("./R/src/wrapper_modify_params.cpp")
#Rcpp::sourceCpp("./R/src/wrapper_model_sdt.cpp")
#Rcpp::sourceCpp("./R/src/wrapper_matrix_prob.cpp")

# ==============================================================================
# 2. 测试全新的 modify_params 功能
# ==============================================================================

cat("=== Test 1: 扁平 List 输入 (默认全部当做 free 参数) ===\n")
res1 <- modify_params(list(d = 2.5))
print(res1$d)          # 预期: 2.5

cat("\n=== Test 2: 降维移动 (使用结构化 List，将默认在 free 的 c_resp 固定到 fixed) ===\n")
res2 <- modify_params(list(fixed = list(c_resp = 1.0)))
print(res2$c_resp)     # 预期: 1.0

cat("\n=== Test 3: 向量序列输入 (自适应转化为 C++ vector) ===\n")
res3 <- modify_params(list(c_conf = c(0.1, 0.5, 0.9, 1.2)))
print(res3$c_conf)     # 预期: 0.1 0.5 0.9 1.2

cat("\n=== Test 4: 乌龙冲突处理 (同参数出现在多槽，高优先级 free 胜出) ===\n")
res4 <- modify_params(list(free = list(rate_decay = 0.8), fixed = list(rate_decay = 0.5)))
print(res4$rate_decay) # 预期: 0.8

cat("\n=== Test 5: 极端情况，传入 NULL (返回默认参数集) ===\n")
res5 <- modify_params(NULL)

print(res5$d)          # 预期: 1.5
print(res5$c_conf)     # 预期: 0.5 1.0 1.5

rm(res1, res2, res3, res4, res5)

# %%
# ==============================================================================
# 3. 测试 ModelSDT 模型 (计算击中率与虚报率)
# ==============================================================================
#Rcpp::sourceCpp("./R/src/wrapper_model_sdt.cpp")

cat("\n=== Test 6: ModelSDT 核心引擎测试 ===\n")
# 获取处理好的完整参数 (设 d=2.0, c_resp=0.5)
std_params <- modify_params(user_params = list(
  free = list(d = 2.0, c_conf = c(0.1, 0.5, 0.9)),
  fixed = list(c_resp = 0)
))

# 直接传入参数，model_sdt 现在会自动读取 c_conf 和 c_resp，智能生成全套积分面积！
res_cdf <- model_sdt(std_params)

cat("False Alarm Rates (所有切点的虚报率):\n", 1 - res_cdf$cdf_noise[[1]], "\n")
cat("Hit Rates (所有切点的击中率):\n", 1 - res_cdf$cdf_signal[[1]], "\n")

# %%
# ==============================================================================
# 4. 测试 matrix_prob 概率矩阵
# ==============================================================================
#Rcpp::sourceCpp("./R/src/wrapper_matrix_prob.cpp")

cat("\n=== Test 7: Matrix Prob 理论概率矩阵 ===\n")
prob_mat <- matrix_prob(res_cdf$cdf_noise, res_cdf$cdf_signal, std_params)

# %%
#Rcpp::sourceCpp("./R/src/wrapper_matrix_mult.cpp")
matrix_mult(freq_mat, prob_mat, std_params)

# %%
#Rcpp::sourceCpp("./R/src/wrapper_criterion_likelihood.cpp")
criterion_likelihood(freq_mat, prob_mat, std_params)

# %% [markdown]
# ## fit

# %%
fit_df <- estimate_mle(
  df = read.csv("data/exp1.csv"), 
  params = list(
    free = list(d = 1.5, c_resp = 0.0, c_conf = c(0.5, 1.0, 1.5)), # 设定我们要找出的自由参数
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "sdt"
)

print(head(fit_df))

# %%
fit_df <- estimate_mle(
  df = read.csv("data/exp3.csv"), 
  colnames = list(
    condition = "FlippedWheel",
    difficulty = "NoiseLevel_Deg"
  ),
  params = list(
    free = list(
      d = c(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
      c_resp = 0.0,
      c_conf = c(0.5, 1.0, 1.5)
    ), # 设定我们要找出的自由参数
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "sdt"
)

print(head(fit_df))

# %%
cat("\n=== Test 12: MCMC NUTS on exp1 ===\n")
fit_mcmc_1 <- estimate_mcmc(
  df = read.csv("data/exp1.csv"),
  params = list(
    free = list(d = 1.5, c_resp = 0.0, c_conf = c(0.5, 1.0, 1.5)),
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "sdt",
  control = list(algorithm = "nuts", samples = 100, warmup = 50, chains = 2)
)

print(head(fit_mcmc_1$fit))

# %%
cat("\n=== Test 13: MCMC NUTS on exp3 ===\n")
fit_mcmc_3 <- estimate_mcmc(
  df = read.csv("data/exp3.csv"),
  colnames = list(
    condition = "FlippedWheel",
    difficulty = "NoiseLevel_Deg"
  ),
  params = list(
    free = list(
      d = c(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
      c_resp = 0.0,
      c_conf = c(0.5, 1.0, 1.5)
    ),
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "sdt",
  control = list(algorithm = "nuts", samples = 100, warmup = 50, chains = 2)
)

print(head(fit_mcmc_3$fit))

# %%
cat("\n=== Test 14: ABC via abcpp on exp1 ===\n")

fit_abc_1 <- estimate_abc(
  df = read.csv("data/exp1.csv"),
  params = list(
    free = list(d = 1.5, c_resp = 0.0, c_conf = c(0.5, 1.0, 1.5)),
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  priors = list(
    d = list(type = "unif", min = 0.2, max = 4.0),
    c_resp = list(type = "norm", mean = 0.0, sd = 0.75),
    c_conf = list(type = "unif", min = 0.2, max = 2.5)
  ),
  model = "sdt",
  control = list(
    method = "rejection",
    tol = 0.1,
    reduction = "none",
    samples = 500,
    seed = 1004
  )
)

print(fit_abc_1$estimator)
print(head(fit_abc_1$fit))

# %%
cat("\n=== Test 15: ABC via abcpp on exp3 ===\n")
abc_control_exp3 <- list(
  method = "rejection",
  tol = 0.1,
  reduction = "none",
  samples = 500,
  seed = 1004
)
abc_params_exp3 <- list(
  free = list(
    d = c(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    c_resp = 0.0,
    c_conf = c(0.5, 1.0, 1.5)
  ),
  fixed = list(sd_signal = 1.0, sd_noise = 1.0)
)
abc_priors_exp3 <- list(
  d = list(type = "unif", min = 0.05, max = 2.5),
  c_resp = list(type = "norm", mean = 0.0, sd = 0.75),
  c_conf = list(type = "unif", min = 0.2, max = 2.5)
)

fit_abc_3 <- estimate_abc(
  df = read.csv("data/exp3.csv"),
  colnames = list(
    condition = "FlippedWheel",
    difficulty = "NoiseLevel_Deg"
  ),
  params = abc_params_exp3,
  model = "sdt",
  control = abc_control_exp3,
  priors = abc_priors_exp3
)

print(fit_abc_3$estimator)
print(fit_abc_3$fit)
