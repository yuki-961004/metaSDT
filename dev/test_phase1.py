import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate

import metaSDT

print("=================================================================")
print("             metaSDT Phase 1 Models Validation Suite              ")
print("=================================================================")

# -----------------------------------------------------------------------------
# 1. Model BCH Test & Verification
# -----------------------------------------------------------------------------
print("\n[1] Testing Model BCH...")
p_resp = 0.5
p_conf = [0.2, 0.4]
d = [1.8]

params_bch = metaSDT.modify_params({
    "d": d,
    "p_resp": p_resp,
    "p_conf": p_conf
})
bch_out = metaSDT.model_bch(params_bch)
print("BCH criteria matrix:", bch_out["criteria"])
print("BCH p_thresholds:", bch_out["p_thresholds"])
print("BCH CDF noise:", bch_out["cdf_noise"])
print("BCH CDF signal:", bch_out["cdf_signal"])

prob_bch_df = metaSDT.matrix_prob(
    cdf_noise=bch_out["cdf_noise"],
    cdf_signal=bch_out["cdf_signal"],
    std_params=params_bch
)
print("BCH Prob DataFrame:\n", prob_bch_df)
assert np.allclose(prob_bch_df.sum(axis=1).values, [1.0, 1.0]), "BCH row sum != 1.0"
print("  -> BCH Matrix Prob OK!")

# -----------------------------------------------------------------------------
# 2. Model Normal Test & Comparison with Scipy Reference
# -----------------------------------------------------------------------------
print("\n[2] Testing Model Normal (Gaussian meta-noise)...")
d_val = 2.0
c_resp = 0.2
c_conf = [0.4, 0.8]
sigma_meta = 0.6

params_normal = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": c_resp,
    "c_conf": c_conf,
    "sigma_meta": sigma_meta
})
normal_out = metaSDT.model_normal(params_normal)
prob_normal = np.array(normal_out["prob_mat"][0])
print("Normal Criteria:", normal_out["criteria"])
print("Normal Prob Matrix:\n", prob_normal)
assert np.allclose(np.sum(prob_normal, axis=1), [1.0, 1.0]), "Normal row sum != 1.0"

# Python/Scipy Reference Integration for Normal
def py_prob_high_normal(mu, c, conf_crit, sigma_meta):
    def integrand(x):
        return stats.norm.pdf(x, loc=mu, scale=1.0) * stats.norm.cdf(x, loc=conf_crit, scale=sigma_meta)
    val, _ = integrate.quad(integrand, c, np.inf)
    return val

# S2 high conf with crit=c_resp + 0.8
mu_s2 = d_val / 2.0
crit_pos2 = c_resp + 0.8
p_high_s2_ref = py_prob_high_normal(mu_s2, c_resp, crit_pos2, sigma_meta)
print("S2 High Conf Rating K (C++):", prob_normal[1, -1], "Reference (Scipy):", p_high_s2_ref)
assert np.isclose(prob_normal[1, -1], p_high_s2_ref, atol=1e-4), "Normal C++ probability does not match reference!"
print("  -> Normal Model Scipy Comparison OK!")

# -----------------------------------------------------------------------------
# 3. Model Lognormal Test & Comparison with Scipy Reference
# -----------------------------------------------------------------------------
print("\n[3] Testing Model Lognormal...")
params_lognormal = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": c_resp,
    "c_conf": c_conf,
    "sigma_meta": sigma_meta
})
lognormal_out = metaSDT.model_lognormal(params_lognormal)
prob_lognormal = np.array(lognormal_out["prob_mat"][0])
print("Lognormal Criteria:", lognormal_out["criteria"])
print("Lognormal Prob Matrix:\n", prob_lognormal)
assert np.allclose(np.sum(prob_lognormal, axis=1), [1.0, 1.0]), "Lognormal row sum != 1.0"

# Python/Scipy Reference Integration for Lognormal
def py_prob_high_lognormal(mu_shifted, conf_log, sigma_meta):
    def integrand(x):
        return stats.norm.pdf(x, loc=mu_shifted, scale=1.0) * stats.norm.cdf(np.log(x), loc=conf_log, scale=sigma_meta)
    val, _ = integrate.quad(integrand, 1e-6, np.inf)
    return val

mu_shifted_s2 = mu_s2 - c_resp
conf_log_pos2 = np.log(0.8)
p_high_logn_ref = py_prob_high_lognormal(mu_shifted_s2, conf_log_pos2, sigma_meta)
print("S2 High Conf Rating K (C++):", prob_lognormal[1, -1], "Reference (Scipy):", p_high_logn_ref)
assert np.isclose(prob_lognormal[1, -1], p_high_logn_ref, atol=1e-4), "Lognormal C++ probability does not match reference!"
print("  -> Lognormal Model Scipy Comparison OK!")

# -----------------------------------------------------------------------------
# 4. Model Decay Test & Comparison with Scipy Reference
# -----------------------------------------------------------------------------
print("\n[4] Testing Model Decay...")
rho_decay = 0.8
params_decay = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": c_resp,
    "c_conf": c_conf,
    "sigma_meta": sigma_meta,
    "rho_decay": [rho_decay]
})
decay_out = metaSDT.model_decay(params_decay)
prob_decay = np.array(decay_out["prob_mat"][0])
print("Decay Criteria:", decay_out["criteria"])
print("Decay Prob Matrix:\n", prob_decay)
assert np.allclose(np.sum(prob_decay, axis=1), [1.0, 1.0]), "Decay row sum != 1.0"

def py_prob_high_decay(mu, c, conf_crit, sigma_meta, delta):
    def integrand(x):
        return stats.norm.pdf(x, loc=mu, scale=1.0) * stats.norm.cdf(delta * x, loc=conf_crit, scale=sigma_meta)
    val, _ = integrate.quad(integrand, c, np.inf)
    return val

p_high_decay_ref = py_prob_high_decay(mu_s2, c_resp, crit_pos2, sigma_meta, rho_decay)
print("S2 High Conf Rating K (C++):", prob_decay[1, -1], "Reference (Scipy):", p_high_decay_ref)
assert np.isclose(prob_decay[1, -1], p_high_decay_ref, atol=1e-4), "Decay C++ probability does not match reference!"
print("  -> Decay Model Scipy Comparison OK!")

# -----------------------------------------------------------------------------
# 5. Fitting Pipelines (MLE, MAP, MCMC, ABC) on Synthetic Data
# -----------------------------------------------------------------------------
print("\n[5] Testing Full Estimator Pipelines (MLE, MAP, MCMC, ABC)...")

# Generate synthetic dataset with shell_run_m
sim_bch = metaSDT.shell_run_m(
    params={"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]},
    model="bch",
    option={"n": 300, "seed": 1004, "plot": False}
)
df_bch = pd.DataFrame(sim_bch["data"])
df_bch["subid"] = 1
print(f"Generated {len(df_bch)} trials for BCH simulation.")

sim_norm = metaSDT.shell_run_m(
    params={"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]},
    model="normal",
    option={"n": 300, "seed": 1004, "plot": False}
)
df_norm = pd.DataFrame(sim_norm["data"])
df_norm["subid"] = 1
print(f"Generated {len(df_norm)} trials for Normal simulation.")

# (A) MLE Fits
print("  -> Running MLE for BCH...")
mle_bch = metaSDT.estimate_mle(
    df=df_bch,
    model="bch",
    params={"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}},
    control={"print_level": 0}
)
print("     MLE BCH Best Params:\n", mle_bch["fit"])

print("  -> Running MLE for Normal...")
mle_norm = metaSDT.estimate_mle(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}},
    control={"print_level": 0}
)
print("     MLE Normal Best Params:\n", mle_norm["fit"])

print("  -> Running MLE for Lognormal...")
mle_logn = metaSDT.estimate_mle(
    df=df_norm,
    model="lognormal",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}},
    control={"print_level": 0}
)
print("     MLE Lognormal Best Params:\n", mle_logn["fit"])

print("  -> Running MLE for Decay...")
mle_decay = metaSDT.estimate_mle(
    df=df_norm,
    model="decay",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "rho_decay": [0.8]}, "fixed": {"c_resp": [0.0], "sigma_meta": [0.2]}},
    control={"print_level": 0}
)
print("     MLE Decay Best Params:\n", mle_decay["fit"])

# (B) MAP Fits
print("  -> Running MAP for Normal...")
priors_map = {
    "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
    "sigma_meta": {"type": "norm", "mean": 0.5, "sd": 1.0}
}
map_norm = metaSDT.estimate_map(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors_map,
    control={"print_level": 0}
)
print("     MAP Normal Best Params:\n", map_norm["fit"])

# (C) MCMC Fits (NUTS)
print("  -> Running MCMC (NUTS) for SDT...")
priors_mcmc = {
    "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
    "c_conf": {"type": "norm", "mean": 0.5, "sd": 1.0}
}
mcmc_sdt = metaSDT.estimate_mcmc(
    df=df_norm,
    model="sdt",
    params={"free": {"d": [1.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors_mcmc,
    control={"sampler": "nuts", "samples": 30, "warmup": 30, "chains": 1, "print_level": 0}
)
print("     MCMC SDT Summary:\n", mcmc_sdt["fit"])

# (D) ABC Fits
print("  -> Running ABC for Normal...")
priors_abc = {
    "d": {"type": "unif", "min": 0.5, "max": 3.0},
    "sigma_meta": {"type": "unif", "min": 0.1, "max": 2.0}
}
abc_norm = metaSDT.estimate_abc(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors_abc,
    control={"n_samples": 50, "n_posterior": 5, "method": "rejection", "print_level": 0}
)
print("     ABC Normal Best Params:\n", abc_norm["fit"])

print("\n=================================================================")
print("      ALL PHASE 1 MODEL MIGRATION VALIDATIONS PASSED!            ")
print("=================================================================")
