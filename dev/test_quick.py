import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate
import sys

import metaSDT

print("Starting test_quick...", flush=True)

# 1. BCH
print("[1] Testing Model BCH...", flush=True)
p_resp = 0.5
p_conf = [0.2, 0.4]
d = [1.8]

params_bch = metaSDT.modify_params({
    "d": d,
    "p_resp": p_resp,
    "p_conf": p_conf
})
bch_out = metaSDT.model_bch(params_bch)
print("BCH criteria matrix:", bch_out["criteria"], flush=True)

prob_bch_df = metaSDT.matrix_prob(
    cdf_noise=bch_out["cdf_noise"],
    cdf_signal=bch_out["cdf_signal"],
    std_params=params_bch
)
print("BCH Prob DataFrame:\n", prob_bch_df, flush=True)
assert np.allclose(prob_bch_df.sum(axis=1).values, [1.0, 1.0]), "BCH row sum != 1.0"
print("  -> BCH Matrix Prob OK!", flush=True)

# 2. Normal
print("\n[2] Testing Model Normal...", flush=True)
d_val = 2.0
c_resp = 0.2
c_conf = [0.4, 0.8]
sigma_meta = 0.6

params_normal = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": [c_resp],
    "c_conf": c_conf,
    "sigma_meta": [sigma_meta]
})
normal_out = metaSDT.model_normal(params_normal)
prob_normal = np.array(normal_out["prob_mat"][0])
print("Normal Criteria:", normal_out["criteria"], flush=True)
print("Normal Prob Matrix:\n", prob_normal, flush=True)
assert np.allclose(np.sum(prob_normal, axis=1), [1.0, 1.0]), "Normal row sum != 1.0"

# 3. Lognormal
print("\n[3] Testing Model Lognormal...", flush=True)
params_lognormal = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": [c_resp],
    "c_conf": c_conf,
    "sigma_meta": [sigma_meta]
})
lognormal_out = metaSDT.model_lognormal(params_lognormal)
prob_lognormal = np.array(lognormal_out["prob_mat"][0])
print("Lognormal Criteria:", lognormal_out["criteria"], flush=True)
print("Lognormal Prob Matrix:\n", prob_lognormal, flush=True)
assert np.allclose(np.sum(prob_lognormal, axis=1), [1.0, 1.0]), "Lognormal row sum != 1.0"

# 4. Decay
print("\n[4] Testing Model Decay...", flush=True)
rho_decay = 0.8
params_decay = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": [c_resp],
    "c_conf": c_conf,
    "sigma_meta": [sigma_meta],
    "rho_decay": [rho_decay]
})
decay_out = metaSDT.model_decay(params_decay)
prob_decay = np.array(decay_out["prob_mat"][0])
print("Decay Criteria:", decay_out["criteria"], flush=True)
print("Decay Prob Matrix:\n", prob_decay, flush=True)
assert np.allclose(np.sum(prob_decay, axis=1), [1.0, 1.0]), "Decay row sum != 1.0"

# 5. Simulations & Fits
print("\n[5] Testing shell_run_m with plot=False...", flush=True)
sim_bch = metaSDT.shell_run_m(
    params={"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]},
    model="bch",
    option={"n": 200, "seed": 1004, "plot": False}
)
df_bch = sim_bch["data"].copy()
df_bch["sub"] = 1

sim_norm = metaSDT.shell_run_m(
    params={"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]},
    model="normal",
    option={"n": 200, "seed": 1004, "plot": False}
)
df_norm = sim_norm["data"].copy()
df_norm["sub"] = 1

print("\nRunning MLE BCH...", flush=True)
mle_bch = metaSDT.estimate_mle(
    df=df_bch,
    model="bch",
    params={"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}},
    control={"print_level": 0, "maxeval": 100}
)
print("MLE BCH result:\n", mle_bch, flush=True)

print("\nRunning MLE Normal...", flush=True)
mle_norm = metaSDT.estimate_mle(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}},
    control={"print_level": 0, "maxeval": 100}
)
print("MLE Normal result:\n", mle_norm, flush=True)

print("\nRunning MLE Lognormal...", flush=True)
mle_logn = metaSDT.estimate_mle(
    df=df_norm,
    model="lognormal",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}},
    control={"print_level": 0, "maxeval": 100}
)
print("MLE Lognormal result:\n", mle_logn, flush=True)

print("\nRunning MLE Decay...", flush=True)
mle_decay = metaSDT.estimate_mle(
    df=df_norm,
    model="decay",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "rho_decay": [0.8]}, "fixed": {"c_resp": [0.0], "sigma_meta": [0.2]}},
    control={"print_level": 0, "maxeval": 100}
)
print("MLE Decay result:\n", mle_decay, flush=True)

print("\nRunning MAP Normal...", flush=True)
priors = {
    "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
    "sigma_meta": {"type": "exp", "rate": 2.0}
}
map_norm = metaSDT.estimate_map(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors,
    control={"print_level": 0, "maxeval": 100}
)
print("MAP Normal result:\n", map_norm, flush=True)

print("\nRunning MCMC (NUTS) Normal...", flush=True)
mcmc_norm = metaSDT.estimate_mcmc(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors,
    control={"sampler": "nuts", "samples": 30, "warmup": 30, "chains": 1, "print_level": 0}
)
print("MCMC Normal result:\n", mcmc_norm, flush=True)

print("\nRunning ABC Normal...", flush=True)
abc_norm = metaSDT.estimate_abc(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors,
    control={"n_samples": 50, "n_posterior": 10, "method": "rejection", "print_level": 0}
)
print("ABC Normal result:\n", abc_norm, flush=True)

print("\n=================================================================", flush=True)
print("      ALL PHASE 1 MODEL MIGRATION VALIDATIONS PASSED!            ", flush=True)
print("=================================================================", flush=True)
