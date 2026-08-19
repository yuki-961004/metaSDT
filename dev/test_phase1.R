# ==============================================================================
#                 metaSDT Phase 1 Model Migration (R Suite)
# ==============================================================================

cat("Loading metaSDT R package...\n")
devtools::load_all("./R")

capture_fit_eval <- function(expr) {
  out <- capture.output({
    msg <- capture.output({
      res <- expr
    }, type = "message")
  })
  combined <- paste(c(out, msg), collapse = "\n")
  prohibited <- c("Fatal", "Dimension mismatch", "NLOPT Error", "Initialization Error")
  for (bad in prohibited) {
    if (grepl(bad, combined, fixed = TRUE)) {
      stop(sprintf("Prohibited diagnostic message '%s' detected in R output:\n%s", bad, combined))
    }
  }
  return(res)
}

# -----------------------------------------------------------------------------
# 0. Test Column Role Resolution (Repair A)
# -----------------------------------------------------------------------------
cat("\n[0] Testing Column Role Resolution in R...\n")
raw_cols_df <- data.frame(
  trial = c(1.0, 2.0, 3.0, 4.0),
  stim = c(0.0, 1.0, 0.0, 1.0),
  resp = c(0.0, 1.0, 0.0, 1.0),
  conf = c(1.0, 2.0, 2.0, 1.0),
  diff = c(1.0, 1.0, 1.0, 1.0),
  evidence = c(-1.234, 2.345, -0.456, 1.890),
  subid = c(101.0, 101.0, 101.0, 101.0)
)
info_res <- info_data(raw_cols_df)
resolved_colnames <- info_res$colnames
print(resolved_colnames)
stopifnot(resolved_colnames$subid == "subid")
stopifnot(resolved_colnames$stim == "stim")
stopifnot(resolved_colnames$resp == "resp")
stopifnot(resolved_colnames$conf == "conf")
stopifnot(resolved_colnames$diff == "diff")
stopifnot("101" %in% names(info_res$subjects))
cat("  -> Column role resolution test passed in R!\n")

# -----------------------------------------------------------------------------
# 1. Model BCH Area, Scalar Consistency & Negative Bounds Verification
# -----------------------------------------------------------------------------
cat("\n[1] Testing Model BCH in R (Scalar Area Architecture & Negative Bounds)...\n")
params_bch <- modify_params(list(
  d = 1.8,
  p_resp = 0.5,
  p_conf = c(0.2, 0.4)
))
bch_out <- model_bch(params_bch)
prob_bch <- matrix_prob(bch_out$cdf_noise, bch_out$cdf_signal, params_bch)
print(prob_bch)

# Blocking Defect C: Check probability matrix row sums equals 1.0 within 1e-10
for (s in 1:2) {
  row_sum <- sum(prob_bch[[1]][s, ])
  stopifnot(abs(row_sum - 1.0) < 1e-10)
}
cat("  -> BCH Matrix Prob in R (1e-10 precision): OK!\n")

# Blocking Defect A check: negative user lower bound normalization and conflict detection
cat("  -> Testing BCH negative user lower bound handling in R...\n")
dummy_bch_df <- data.frame(
  stim = rep(c(0.0, 0.0, 1.0, 1.0), 50),
  resp = rep(c(0.0, 1.0, 0.0, 1.0), 50),
  conf = rep(c(1.0, 2.0, 1.0, 2.0), 50),
  subid = rep(1.0, 200)
)
mle_neg_bound <- capture_fit_eval(estimate_mle(
  df = dummy_bch_df,
  model = "bch",
  params = list(free = list(d = 1.5, p_conf = c(0.2, 0.4)), fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0)),
  lower = list(d = -2.0),
  control = list(print_level = 0)
))
stopifnot(mle_neg_bound$fit$status[1] >= 0)
stopifnot(mle_neg_bound$fit$d[1] > 0)
cat(sprintf("     BCH MLE with negative user lower bound passed without error (d = %.4f)\n", mle_neg_bound$fit$d[1]))

domain_error_caught <- FALSE
tryCatch({
  estimate_mle(
    df = dummy_bch_df,
    model = "bch",
    params = list(free = list(d = 1.5, p_conf = c(0.2, 0.4)), fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0)),
    lower = list(d = 5.0),
    upper = list(d = 2.0),
    control = list(print_level = 0)
  )
}, error = function(e) {
  domain_error_caught <<- TRUE
  cat(sprintf("     Boundary conflict correctly caught: %s\n", conditionMessage(e)))
})
stopifnot(domain_error_caught)

# -----------------------------------------------------------------------------
# 2. Model Normal Test
# -----------------------------------------------------------------------------
cat("\n[2] Testing Model Normal in R...\n")
params_norm <- modify_params(list(
  d = 1.8,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6
))
norm_out <- model_normal(params_norm)
print(norm_out$prob_mat[[1]])
stopifnot(abs(sum(norm_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(norm_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Normal Model in R: OK!\n")

# -----------------------------------------------------------------------------
# 3. Model Lognormal Test
# -----------------------------------------------------------------------------
cat("\n[3] Testing Model Lognormal in R...\n")
params_logn <- modify_params(list(
  d = 1.8,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6
))
logn_out <- model_lognormal(params_logn)
print(logn_out$prob_mat[[1]])
stopifnot(abs(sum(logn_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(logn_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Lognormal Model in R: OK!\n")

# -----------------------------------------------------------------------------
# 4. Model Decay Test
# -----------------------------------------------------------------------------
cat("\n[4] Testing Model Decay in R...\n")
params_decay <- modify_params(list(
  d = 1.8,
  c_resp = 0.2,
  c_conf = c(0.4, 0.8),
  sigma_meta = 0.6,
  rho_decay = 0.8
))
decay_out <- model_decay(params_decay)
print(decay_out$prob_mat[[1]])
stopifnot(abs(sum(decay_out$prob_mat[[1]][[1]]) - 1.0) < 1e-4)
stopifnot(abs(sum(decay_out$prob_mat[[1]][[2]]) - 1.0) < 1e-4)
cat("  -> Decay Model in R: OK!\n")

# -----------------------------------------------------------------------------
# 5. Full 4-Model x 4-Estimator Validation Matrix with Shape Assertion & Diagnostics
# -----------------------------------------------------------------------------
cat("\n[5] Running 4-Model x 4-Estimator Validation Matrix in R...\n")

models <- c("bch", "normal", "lognormal", "decay")
test_data <- list()

for (m in models) {
  if (m == "bch") {
    sim <- shell_run_m(
      params = list(
        free = list(d = 1.5, p_conf = c(0.2, 0.4)),
        fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0)
      ),
      model = "bch",
      option = list(n = 300, seed = 1004, plot = FALSE)
    )
  } else if (m == "decay") {
    sim <- shell_run_m(
      params = list(
        free = list(d = 1.5, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8),
        fixed = list(c_resp = 0.0, sd_signal = 1.0, sd_noise = 1.0)
      ),
      model = "decay",
      option = list(n = 300, seed = 1004, plot = FALSE)
    )
  } else {
    sim <- shell_run_m(
      params = list(
        free = list(d = 1.5, c_conf = c(0.4, 0.8), sigma_meta = 0.5),
        fixed = list(c_resp = 0.0, sd_signal = 1.0, sd_noise = 1.0)
      ),
      model = m,
      option = list(n = 300, seed = 1004, plot = FALSE)
    )
  }
  df <- sim$data
  df$subid <- 1.0
  test_data[[m]] <- df
  cat(sprintf("  -> Generated %d trials for model '%s'.\n", nrow(df), m))
}

verify_shape_agreement_r <- function(df, model_name, params_list) {
  mod_params <- modify_params(params_list)
  freq_obj <- matrix_freq(
    stim = df$stim,
    resp = df$resp,
    conf = if ("conf" %in% names(df)) df$conf else NULL,
    diff = if ("diff" %in% names(df)) df$diff else NULL
  )
  if (model_name == "bch") {
    bch_eval <- model_bch(mod_params)
    prob_obj <- matrix_prob(bch_eval$cdf_noise, bch_eval$cdf_signal, mod_params)
    prob_mat <- prob_obj[[1]]
  } else if (model_name == "normal") {
    prob_mat <- do.call(rbind, model_normal(mod_params)$prob_mat[[1]])
  } else if (model_name == "lognormal") {
    prob_mat <- do.call(rbind, model_lognormal(mod_params)$prob_mat[[1]])
  } else if (model_name == "decay") {
    prob_mat <- do.call(rbind, model_decay(mod_params)$prob_mat[[1]])
  }

  freq_mat <- freq_obj[[1]]
  stopifnot(identical(dim(freq_mat), dim(prob_mat)))
  if (model_name == "bch") {
    stopifnot(identical(dim(freq_mat), as.integer(c(2, 6))))
  }
  return(dim(freq_mat))
}

assert_fit_success <- function(res, model_name, estimator_name) {
  fit_df <- res$fit
  stopifnot(nrow(fit_df) == 1)
  stopifnot(fit_df$subid[1] == 1.0)
  status <- fit_df$status[1]
  stopifnot(!is.na(status) && status >= 0)
  stopifnot(is.finite(fit_df$d[1]))
  for (col in names(fit_df)) {
    val <- fit_df[[col]][1]
    if (is.numeric(val) && !is.na(val)) {
      stopifnot(is.finite(val))
    }
  }
}

# (A) MLE Fits
cat("\n--- Testing MLE across all 4 models in R ---\n")
for (m in models) {
  cat(sprintf("  -> Running MLE for '%s' in R...\n", m))
  if (m == "bch") {
    p <- list(free = list(d = 1.5, p_conf = c(0.2, 0.4)), fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, p_resp = 0.5, p_conf = c(0.2, 0.4)))
  } else if (m == "decay") {
    p <- list(free = list(d = 1.5, c_conf = c(0.4, 0.8), rho_decay = 0.8), fixed = list(c_resp = 0.0, sigma_meta = 0.3, sd_signal = 1.0, sd_noise = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8))
  } else {
    p <- list(free = list(d = 1.5, c_conf = c(0.4, 0.8), sigma_meta = 0.5), fixed = list(c_resp = 0.0, sd_signal = 1.0, sd_noise = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5))
  }
  mle_res <- capture_fit_eval(estimate_mle(
    df = test_data[[m]],
    model = m,
    params = p,
    control = list(print_level = 0)
  ))
  assert_fit_success(mle_res, m, "MLE")
  cat(sprintf("     MLE '%s' Success: d = %.4f, shape = [%d x %d]\n", m, mle_res$fit$d[1], sh[1], sh[2]))
}

# (B) MAP Fits
cat("\n--- Testing MAP across all 4 models in R ---\n")
for (m in models) {
  cat(sprintf("  -> Running MAP for '%s' in R...\n", m))
  if (m == "bch") {
    p <- list(free = list(d = 1.5, p_conf = c(0.2, 0.4)), fixed = list(p_resp = 0.5, sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(
      d = list(type = "norm", mean = 1.5, sd = 1.0),
      p_conf = list(type = "unif", min = 0.05, max = 0.45)
    )
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, p_resp = 0.5, p_conf = c(0.2, 0.4)))
  } else if (m == "decay") {
    p <- list(free = list(d = 1.5, rho_decay = 0.8), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(
      d = list(type = "norm", mean = 1.5, sd = 1.0),
      rho_decay = list(type = "unif", min = 0.1, max = 1.0)
    )
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8))
  } else {
    p <- list(free = list(d = 1.5, sigma_meta = 0.5), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(
      d = list(type = "norm", mean = 1.5, sd = 1.0),
      sigma_meta = list(type = "norm", mean = 0.5, sd = 1.0)
    )
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5))
  }
  map_res <- capture_fit_eval(estimate_map(
    df = test_data[[m]],
    model = m,
    params = p,
    priors = priors,
    control = list(print_level = 0)
  ))
  assert_fit_success(map_res, m, "MAP")
  cat(sprintf("     MAP '%s' Success: d = %.4f, shape = [%d x %d]\n", m, map_res$fit$d[1], sh[1], sh[2]))
}

# (C) MCMC Fits
cat("\n--- Testing MCMC across all 4 models in R ---\n")
for (m in models) {
  cat(sprintf("  -> Running MCMC for '%s' in R...\n", m))
  if (m == "bch") {
    p <- list(free = list(d = 1.5), fixed = list(p_resp = 0.5, p_conf = c(0.2, 0.4), sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(d = list(type = "norm", mean = 1.5, sd = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, p_resp = 0.5, p_conf = c(0.2, 0.4)))
  } else if (m == "decay") {
    p <- list(free = list(d = 1.5), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8, sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(d = list(type = "norm", mean = 1.5, sd = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8))
  } else {
    p <- list(free = list(d = 1.5), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5, sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(d = list(type = "norm", mean = 1.5, sd = 1.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5))
  }
  mcmc_res <- capture_fit_eval(estimate_mcmc(
    df = test_data[[m]],
    model = m,
    params = p,
    priors = priors,
    control = list(sampler = "nuts", samples = 20, warmup = 20, chains = 1, print_level = 0)
  ))
  assert_fit_success(mcmc_res, m, "MCMC")
  cat(sprintf("     MCMC '%s' Success: d = %.4f, shape = [%d x %d]\n", m, mcmc_res$fit$d[1], sh[1], sh[2]))
}

# (D) ABC Fits
cat("\n--- Testing ABC across all 4 models in R ---\n")
for (m in models) {
  cat(sprintf("  -> Running ABC for '%s' in R...\n", m))
  if (m == "bch") {
    p <- list(free = list(d = 1.5), fixed = list(p_resp = 0.5, p_conf = c(0.2, 0.4), sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(d = list(type = "unif", min = 0.5, max = 3.0))
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, p_resp = 0.5, p_conf = c(0.2, 0.4)))
  } else if (m == "decay") {
    p <- list(free = list(d = 1.5, rho_decay = 0.8), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(
      d = list(type = "unif", min = 0.5, max = 3.0),
      rho_decay = list(type = "unif", min = 0.1, max = 1.0)
    )
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.3, rho_decay = 0.8))
  } else {
    p <- list(free = list(d = 1.5, sigma_meta = 0.5), fixed = list(c_resp = 0.0, c_conf = c(0.4, 0.8), sd_signal = 1.0, sd_noise = 1.0))
    priors <- list(
      d = list(type = "unif", min = 0.5, max = 3.0),
      sigma_meta = list(type = "unif", min = 0.1, max = 2.0)
    )
    sh <- verify_shape_agreement_r(test_data[[m]], m, list(d = 1.5, c_resp = 0.0, c_conf = c(0.4, 0.8), sigma_meta = 0.5))
  }
  abc_res <- capture_fit_eval(estimate_abc(
    df = test_data[[m]],
    model = m,
    params = p,
    priors = priors,
    control = list(samples = 40, n_posterior = 5, method = "rejection", print_level = 0)
  ))
  assert_fit_success(abc_res, m, "ABC")
  cat(sprintf("     ABC '%s' Success: d = %.4f, shape = [%d x %d]\n", m, abc_res$fit$d[1], sh[1], sh[2]))
}

cat("\n=================================================================\n")
cat("      ALL PHASE 1 R VALIDATIONS COMPLETED SUCCESSFULLY!          \n")
cat("=================================================================\n")
