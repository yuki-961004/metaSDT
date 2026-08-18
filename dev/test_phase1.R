# ==============================================================================
#                 metaSDT Phase 1 Model Migration (R Suite)
# ==============================================================================

# 1. 编译并加载 R 包
cat("Loading metaSDT R package...\n")
devtools::load_all("./R")

cat("\n=================================================================\n")
cat("                Testing Model BCH in R\n")
cat("=================================================================\n")
params_bch <- modify_params(list(
  d = 1.8,
  p_resp = 0.5,
  p_conf = c(0.2, 0.4)
))
bch_out <- model_bch(params_bch)
cat("BCH criteria:\n")
print(bch_out$criteria)
cat("BCH p_thresholds:\n")
print(bch_out$p_thresholds)

prob_bch <- matrix_prob(bch_out$cdf_noise, bch_out$cdf_signal, params_bch)
cat("BCH Probability Matrix:\n")
print(prob_bch)
stopifnot(all(abs(rowSums(prob_bch[[1]]) - 1.0) < 1e-4))
cat("  -> BCH Matrix Prob in R: OK!\n")

cat("\n=================================================================\n")
cat("                Testing Model Normal in R\n")
cat("=================================================================\n")
params_norm <- modify_params(list(
  d = 2.0,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6
))
norm_out <- model_normal(params_norm)
cat("Normal criteria:\n")
print(norm_out$criteria)
cat("Normal Prob Matrix:\n")
print(norm_out$prob_mat[[1]])
stopifnot(abs(sum(norm_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(norm_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Normal Model in R: OK!\n")

cat("\n=================================================================\n")
cat("                Testing Model Lognormal in R\n")
cat("=================================================================\n")
params_logn <- modify_params(list(
  d = 2.0,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6
))
logn_out <- model_lognormal(params_logn)
cat("Lognormal criteria:\n")
print(logn_out$criteria)
cat("Lognormal Prob Matrix:\n")
print(logn_out$prob_mat[[1]])
stopifnot(abs(sum(logn_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(logn_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Lognormal Model in R: OK!\n")

cat("\n=================================================================\n")
cat("                Testing Model Decay in R\n")
cat("=================================================================\n")
params_decay <- modify_params(list(
  d = 2.0,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6,
  rho_decay = 0.8
))
decay_out <- model_decay(params_decay)
cat("Decay criteria:\n")
print(decay_out$criteria)
cat("Decay Prob Matrix:\n")
print(decay_out$prob_mat[[1]])
stopifnot(abs(sum(decay_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(decay_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Decay Model in R: OK!\n")

cat("\n=================================================================\n")
cat("         Testing Simulations (shell_run_m) & Fitting in R\n")
cat("=================================================================\n")
sim_norm <- shell_run_m(
  params = list(
    free = list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5),
    fixed = list(sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "normal",
  option = list(n = 300, seed = 1004, plot = FALSE)
)
df_norm <- sim_norm$data
cat("Generated simulated Normal trials in R:\n")
print(head(df_norm))

cat("\n[1] Running MLE for Normal in R...\n")
fit_mle_norm <- estimate_mle(
  df = df_norm,
  model = "normal",
  params = list(
    free = list(d = 1.5, c_conf = c(0.4, 0.8), sigma_meta = 0.5),
    fixed = list(c_resp = 0.0, sd_signal = 1.0, sd_noise = 1.0)
  ),
  control = list(print_level = 0)
)
print(fit_mle_norm$fit)

cat("\n[2] Running MLE for BCH in R...\n")
sim_bch <- shell_run_m(
  params = list(
    free = list(d = 1.5, p_conf = c(0.2, 0.4)),
    fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0)
  ),
  model = "bch",
  option = list(n = 300, seed = 1004, plot = FALSE)
)
fit_mle_bch <- estimate_mle(
  df = sim_bch$data,
  model = "bch",
  params = list(
    free = list(d = 1.5, p_conf = c(0.2, 0.4)),
    fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0)
  ),
  control = list(print_level = 0)
)
print(fit_mle_bch$fit)

cat("\n[3] Running MAP for Normal in R...\n")
priors_map <- list(
  d = list(type = "norm", mean = 1.5, sd = 1.0),
  sigma_meta = list(type = "norm", mean = 0.5, sd = 1.0)
)
fit_map_norm <- estimate_map(
  df = df_norm,
  model = "normal",
  params = list(
    free = list(d = 1.5, sigma_meta = 0.5),
    fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sd_signal = 1.0, sd_noise = 1.0)
  ),
  priors = priors_map,
  control = list(print_level = 0)
)
print(fit_map_norm$fit)

cat("\n[4] Running ABC for Normal in R...\n")
priors_abc <- list(
  d = list(type = "unif", min = 0.5, max = 3.0),
  sigma_meta = list(type = "unif", min = 0.1, max = 2.0)
)
fit_abc_norm <- estimate_abc(
  df = df_norm,
  model = "normal",
  params = list(
    free = list(d = 1.5, sigma_meta = 0.5),
    fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sd_signal = 1.0, sd_noise = 1.0)
  ),
  priors = priors_abc,
  control = list(samples = 50, n_posterior = 5, method = "rejection", print_level = 0)
)
print(fit_abc_norm$fit)

cat("\n=================================================================\n")
cat("      ALL PHASE 1 R VALIDATIONS COMPLETED SUCCESSFULLY!          \n")
cat("=================================================================\n")
