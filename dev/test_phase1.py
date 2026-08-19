import os
import sys
import tempfile
from contextlib import contextmanager

# Ensure local Python/ folder is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Python")))

import metaSDT
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate

print("=================================================================")
print("             metaSDT Phase 1 Models Validation Suite              ")
print("=================================================================")

@contextmanager
def capture_diagnostics():
    """Captures OS fd 2 (stderr) to detect any backend fatal errors or warnings."""
    sys.stderr.flush()
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    tfile = tempfile.TemporaryFile(mode='w+b')
    os.dup2(tfile.fileno(), stderr_fd)
    try:
        yield tfile
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        tfile.seek(0)
        output = tfile.read().decode('utf-8', errors='replace')
        tfile.close()

        # Treat any fatal error, dimension mismatch, or NLopt crash as immediate test failure
        prohibited = ["Fatal", "Dimension mismatch", "NLOPT Error", "Initialization Error"]
        for bad in prohibited:
            if bad in output:
                raise AssertionError(f"Prohibited diagnostic message '{bad}' detected in backend output:\n{output}")

# -----------------------------------------------------------------------------
# 0. Test Column Role Resolution (Repair A)
# -----------------------------------------------------------------------------
print("\n[0] Testing Column Role Resolution...")
raw_cols_df = {
    "trial": [1.0, 2.0, 3.0, 4.0],
    "stim": [0.0, 1.0, 0.0, 1.0],
    "resp": [0.0, 1.0, 0.0, 1.0],
    "conf": [1.0, 2.0, 2.0, 1.0],
    "diff": [1.0, 1.0, 1.0, 1.0],
    "evidence": [-1.234, 2.345, -0.456, 1.890],
    "subid": [101.0, 101.0, 101.0, 101.0]
}
info_res = metaSDT.info_data(pd.DataFrame(raw_cols_df))
resolved_colnames = info_res["colnames"]
print("Resolved colnames:", resolved_colnames)
assert resolved_colnames["subid"] == "subid", "Role 'subid' must resolve to column 'subid'!"
assert resolved_colnames["stim"] == "stim", "Role 'stim' must resolve to column 'stim'!"
assert resolved_colnames["resp"] == "resp", "Role 'resp' must resolve to column 'resp'!"
assert resolved_colnames["conf"] == "conf", "Role 'conf' must resolve to column 'conf'!"
assert resolved_colnames["diff"] == "diff", "Role 'diff' must resolve to column 'diff'!"
assert "101" in info_res["subjects"], "Subject '101' must be extracted in subjects dictionary!"
print("  -> Column role resolution test passed!")

# -----------------------------------------------------------------------------
# 1. Model BCH Area, Scalar Consistency & Negative Bounds Verification
# -----------------------------------------------------------------------------
print("\n[1] Testing Model BCH (Scalar Area Architecture & Negative Bounds)...")
p_resp = 0.5
p_conf = [0.2, 0.4]
d_val = 1.8

params_bch = metaSDT.modify_params({
    "d": [d_val],
    "p_resp": p_resp,
    "p_conf": p_conf
})
bch_out = metaSDT.model_bch(params_bch)
prob_bch_df = metaSDT.matrix_prob(
    cdf_noise=bch_out["cdf_noise"],
    cdf_signal=bch_out["cdf_signal"],
    std_params=params_bch
)
print("BCH criteria matrix:", bch_out["criteria"])
print("BCH p_thresholds:", bch_out["p_thresholds"])
print("BCH Prob DataFrame:\n", prob_bch_df)

# Blocking Defect C check: check probability matrix row sums equals 1.0 to 1e-10
bch_prob_mat = prob_bch_df.values

# Direct scalar area check using ModelBCH.area(...)
model_bch_obj = metaSDT._model_bch.ModelBCH(params_bch)
crit = bch_out["criteria"][0]
bounds = [-np.inf, crit[0], crit[1], crit[2], crit[3], crit[4], np.inf]

for stim_idx in range(2):
    scalar_probs = []
    for cell_idx in range(6):
        resp_idx = 0 if cell_idx < 3 else 1
        low = bounds[cell_idx]
        upp = bounds[cell_idx + 1]
        p_cell = model_bch_obj.area(stim_idx, resp_idx, low, upp, 0)
        scalar_probs.append(p_cell)
        # Assert scalar area matches exported probability matrix cell to 1e-10
        assert abs(p_cell - bch_prob_mat[stim_idx, cell_idx]) < 1e-10, \
            f"Scalar area mismatch at stim {stim_idx}, cell {cell_idx}: {p_cell} vs {bch_prob_mat[stim_idx, cell_idx]}"

    assert abs(sum(scalar_probs) - 1.0) < 1e-10, f"Scalar probs sum for stim {stim_idx} != 1.0"

print("  -> BCH Matrix Prob & Scalar Area Equality to 1e-10 OK!")

# Blocking Defect A check: negative user lower bound normalization and conflict detection
print("  -> Testing BCH negative user lower bound handling...")
dummy_bch_df = pd.DataFrame({
    "stim": [0.0, 0.0, 1.0, 1.0] * 50,
    "resp": [0.0, 1.0, 0.0, 1.0] * 50,
    "conf": [1.0, 2.0, 1.0, 2.0] * 50,
    "subid": [1.0] * 200
})
with capture_diagnostics():
    mle_neg_bound = metaSDT.estimate_mle(
        df=dummy_bch_df,
        model="bch",
        params={"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}},
        lower={"d": [-2.0]},
        control={"print_level": 0}
    )
assert mle_neg_bound["fit"]["status"].iloc[0] >= 0, "BCH MLE with custom negative lower bound failed!"
assert mle_neg_bound["fit"]["d"].iloc[0] > 0, "Estimated BCH sensitivity must be strictly positive!"
print("     BCH MLE with negative user lower bound passed without error (d =", mle_neg_bound["fit"]["d"].iloc[0], ")")

# Assert invalid domain (lower >= upper) raises exception before fitting
domain_error_caught = False
try:
    metaSDT.estimate_mle(
        df=dummy_bch_df,
        model="bch",
        params={"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}},
        lower={"d": [5.0]},
        upper={"d": [2.0]},
        control={"print_level": 0}
    )
except Exception as e:
    domain_error_caught = True
    print(f"     Boundary conflict correctly caught: {e}")
assert domain_error_caught, "Expected exception for invalid domain lower_bound >= upper_bound!"

# -----------------------------------------------------------------------------
# 2. Model Normal Test & Comparison with Scipy Reference
# -----------------------------------------------------------------------------
print("\n[2] Testing Model Normal (Gaussian meta-noise)...")
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

def py_prob_high_normal(mu, c, conf_crit, sigma_meta):
    def integrand(x):
        return stats.norm.pdf(x, loc=mu, scale=1.0) * stats.norm.cdf(x, loc=conf_crit, scale=sigma_meta)
    val, _ = integrate.quad(integrand, c, np.inf)
    return val

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
# 5. Full 4-Model x 4-Estimator Validation Matrix with Shape Assertion & Stderr Capture
# -----------------------------------------------------------------------------
print("\n[5] Running 4-Model x 4-Estimator Validation Matrix...")

models = ["bch", "normal", "lognormal", "decay"]
test_data = {}

for m in models:
    if m == "bch":
        sim = metaSDT.shell_run_m(
            params={"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]},
            model="bch",
            option={"n": 300, "seed": 1004, "plot": False}
        )
    elif m == "decay":
        sim = metaSDT.shell_run_m(
            params={"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]},
            model="decay",
            option={"n": 300, "seed": 1004, "plot": False}
        )
    else:
        sim = metaSDT.shell_run_m(
            params={"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]},
            model=m,
            option={"n": 300, "seed": 1004, "plot": False}
        )
    df = pd.DataFrame(sim["data"])
    df["subid"] = 1.0
    test_data[m] = df
    print(f"  -> Generated {len(df)} trials for model '{m}'. Columns: {list(df.columns)}")

def verify_shape_agreement(df, model_name, params_dict):
    """Independently construct observed MatrixFreq and model MatrixProb to assert exact shape equality."""
    mod_params = metaSDT.modify_params(params_dict)
    freq_df = metaSDT.matrix_freq(
        stim=df["stim"].tolist(),
        resp=df["resp"].tolist(),
        conf=df["conf"].tolist(),
        diff=df["diff"].tolist() if "diff" in df.columns else None,
        std_params=mod_params
    )
    if model_name == "bch":
        bch_eval = metaSDT.model_bch(mod_params)
        prob_df = metaSDT.matrix_prob(
            cdf_noise=bch_eval["cdf_noise"],
            cdf_signal=bch_eval["cdf_signal"],
            std_params=mod_params
        )
        prob_shape = prob_df.shape
    elif model_name == "normal":
        prob_shape = np.array(metaSDT.model_normal(mod_params)["prob_mat"][0]).shape
    elif model_name == "lognormal":
        prob_shape = np.array(metaSDT.model_lognormal(mod_params)["prob_mat"][0]).shape
    elif model_name == "decay":
        prob_shape = np.array(metaSDT.model_decay(mod_params)["prob_mat"][0]).shape

    freq_shape = freq_df.shape
    assert freq_shape == prob_shape, f"[{model_name}] Shape mismatch before fit: freq={freq_shape}, prob={prob_shape}"
    if model_name == "bch":
        assert freq_shape == (2, 6), f"[bch] Expected shape (2, 6), got {freq_shape}"
    return freq_shape

def assert_fit_success(res, model_name, estimator_name):
    fit = res["fit"]
    if isinstance(fit, dict):
        fit_df = pd.DataFrame(fit)
    else:
        fit_df = fit

    assert len(fit_df) == 1, f"[{model_name} {estimator_name}] Expected 1 subject, got {len(fit_df)}"
    assert fit_df["subid"].iloc[0] == 1.0, f"[{model_name} {estimator_name}] Expected subid=1.0, got {fit_df['subid'].iloc[0]}"

    status = fit_df["status"].iloc[0]
    assert status >= 0, f"[{model_name} {estimator_name}] Fit status indicates failure: {status}"

    for col in fit_df.columns:
        val = fit_df[col].iloc[0]
        if isinstance(val, (int, float, np.number)):
            assert np.isfinite(val), f"[{model_name} {estimator_name}] Non-finite value in col '{col}': {val}"

# (A) MLE Fits
print("\n--- Testing MLE across all 4 models ---")
for m in models:
    print(f"  -> Running MLE for '{m}'...")
    if m == "bch":
        p = {"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]})
    elif m == "decay":
        p = {"free": {"d": [1.5], "c_conf": [0.4, 0.8], "rho_decay": [0.8]}, "fixed": {"c_resp": [0.0], "sigma_meta": [0.3]}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]})
    else:
        p = {"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]})

    with capture_diagnostics():
        mle_res = metaSDT.estimate_mle(
            df=test_data[m],
            model=m,
            params=p,
            control={"print_level": 0}
        )
    assert_fit_success(mle_res, m, "MLE")
    print(f"     MLE '{m}' Success: d = {mle_res['fit']['d'].iloc[0]:.4f}, shape = {shape}")

# (B) MAP Fits
print("\n--- Testing MAP across all 4 models ---")
for m in models:
    print(f"  -> Running MAP for '{m}'...")
    if m == "bch":
        p = {"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}}
        priors = {
            "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
            "p_conf": {"type": "unif", "min": 0.05, "max": 0.45}
        }
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]})
    elif m == "decay":
        p = {"free": {"d": [1.5], "rho_decay": [0.8]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3]}}
        priors = {
            "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
            "rho_decay": {"type": "unif", "min": 0.1, "max": 1.0}
        }
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]})
    else:
        p = {"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}}
        priors = {
            "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
            "sigma_meta": {"type": "norm", "mean": 0.5, "sd": 1.0}
        }
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]})

    with capture_diagnostics():
        map_res = metaSDT.estimate_map(
            df=test_data[m],
            model=m,
            params=p,
            priors=priors,
            control={"print_level": 0}
        )
    assert_fit_success(map_res, m, "MAP")
    print(f"     MAP '{m}' Success: d = {map_res['fit']['d'].iloc[0]:.4f}, shape = {shape}")

# (C) MCMC Fits
print("\n--- Testing MCMC across all 4 models ---")
for m in models:
    print(f"  -> Running MCMC for '{m}'...")
    if m == "bch":
        p = {"free": {"d": [1.5]}, "fixed": {"p_resp": [0.5], "p_conf": [0.2, 0.4]}}
        priors = {"d": {"type": "norm", "mean": 1.5, "sd": 1.0}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]})
    elif m == "decay":
        p = {"free": {"d": [1.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]}}
        priors = {"d": {"type": "norm", "mean": 1.5, "sd": 1.0}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]})
    else:
        p = {"free": {"d": [1.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}}
        priors = {"d": {"type": "norm", "mean": 1.5, "sd": 1.0}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]})

    with capture_diagnostics():
        mcmc_res = metaSDT.estimate_mcmc(
            df=test_data[m],
            model=m,
            params=p,
            priors=priors,
            control={"sampler": "nuts", "samples": 20, "warmup": 20, "chains": 1, "print_level": 0}
        )
    assert_fit_success(mcmc_res, m, "MCMC")
    print(f"     MCMC '{m}' Success: d = {mcmc_res['fit']['d'].iloc[0]:.4f}, shape = {shape}")

# (D) ABC Fits
print("\n--- Testing ABC across all 4 models ---")
for m in models:
    print(f"  -> Running ABC for '{m}'...")
    if m == "bch":
        p = {"free": {"d": [1.5]}, "fixed": {"p_resp": [0.5], "p_conf": [0.2, 0.4]}}
        priors = {"d": {"type": "unif", "min": 0.5, "max": 3.0}}
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]})
    elif m == "decay":
        p = {"free": {"d": [1.5], "rho_decay": [0.8]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3]}}
        priors = {
            "d": {"type": "unif", "min": 0.5, "max": 3.0},
            "rho_decay": {"type": "unif", "min": 0.1, "max": 1.0}
        }
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.3], "rho_decay": [0.8]})
    else:
        p = {"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}}
        priors = {
            "d": {"type": "unif", "min": 0.5, "max": 3.0},
            "sigma_meta": {"type": "unif", "min": 0.1, "max": 2.0}
        }
        shape = verify_shape_agreement(test_data[m], m, {"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]})

    with capture_diagnostics():
        abc_res = metaSDT.estimate_abc(
            df=test_data[m],
            model=m,
            params=p,
            priors=priors,
            control={"n_samples": 40, "n_posterior": 5, "method": "rejection", "print_level": 0}
        )
    assert_fit_success(abc_res, m, "ABC")
    print(f"     ABC '{m}' Success: d = {abc_res['fit']['d'].iloc[0]:.4f}, shape = {shape}")

print("\n=================================================================")
print("      ALL PHASE 1 MODEL MIGRATION VALIDATIONS PASSED!            ")
print("=================================================================")
