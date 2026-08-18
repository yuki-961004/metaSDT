import sys
import numpy as np
import pandas as pd
import metaSDT

print("Testing Model BCH...", flush=True)
p_resp = 0.5
p_conf = [0.2, 0.4]
d = [1.8]
params_bch = metaSDT.modify_params({
    "d": d,
    "p_resp": p_resp,
    "p_conf": p_conf
})
bch_out = metaSDT.model_bch(params_bch)
print("BCH OK!", flush=True)

print("Testing Model Normal...", flush=True)
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
print("Normal OK!", flush=True)

print("Testing Model Lognormal...", flush=True)
params_lognormal = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": c_resp,
    "c_conf": c_conf,
    "sigma_meta": sigma_meta
})
lognormal_out = metaSDT.model_lognormal(params_lognormal)
print("Lognormal OK!", flush=True)

print("Testing Model Decay...", flush=True)
params_decay = metaSDT.modify_params({
    "d": [d_val],
    "c_resp": c_resp,
    "c_conf": c_conf,
    "sigma_meta": sigma_meta,
    "rho_decay": [0.8]
})
decay_out = metaSDT.model_decay(params_decay)
print("Decay OK!", flush=True)

print("Testing Simulation BCH...", flush=True)
sim_bch = metaSDT.shell_run_m(
    params={"d": [1.5], "p_resp": [0.5], "p_conf": [0.2, 0.4]},
    model="bch",
    option={"n": 200, "seed": 1004, "plot": False}
)
df_bch = pd.DataFrame(sim_bch["data"])
df_bch["subid"] = 1
print("Simulation BCH OK!", flush=True)

print("Testing Simulation Normal...", flush=True)
sim_norm = metaSDT.shell_run_m(
    params={"d": [1.5], "c_resp": [0.0], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]},
    model="normal",
    option={"n": 200, "seed": 1004, "plot": False}
)
df_norm = pd.DataFrame(sim_norm["data"])
df_norm["subid"] = 1
print("Simulation Normal OK!", flush=True)

print("Testing MLE BCH...", flush=True)
mle_bch = metaSDT.estimate_mle(
    df=df_bch,
    model="bch",
    params={"free": {"d": [1.5], "p_conf": [0.2, 0.4]}, "fixed": {"p_resp": [0.5]}},
    control={"print_level": 0}
)
print("MLE BCH OK:\n", mle_bch, flush=True)

print("Testing MLE Normal...", flush=True)
mle_norm = metaSDT.estimate_mle(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "c_conf": [0.4, 0.8], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0]}},
    control={"print_level": 0}
)
print("MLE Normal OK:\n", mle_norm, flush=True)

print("Testing MAP Normal...", flush=True)
priors = {
    "d": {"type": "norm", "mean": 1.5, "sd": 1.0},
    "sigma_meta": {"type": "norm", "mean": 0.5, "sd": 1.0}
}
map_norm = metaSDT.estimate_map(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=priors,
    control={"print_level": 0}
)
print("MAP Normal OK:\n", map_norm, flush=True)

print("Testing ABC Normal...", flush=True)
abc_priors = {
    "d": {"type": "unif", "min": 0.5, "max": 3.0},
    "sigma_meta": {"type": "unif", "min": 0.1, "max": 2.0}
}
abc_norm = metaSDT.estimate_abc(
    df=df_norm,
    model="normal",
    params={"free": {"d": [1.5], "sigma_meta": [0.5]}, "fixed": {"c_resp": [0.0], "c_conf": [0.4, 0.8]}},
    priors=abc_priors,
    control={"n_samples": 50, "n_posterior": 5, "method": "rejection", "print_level": 0}
)
print("ABC Normal OK:\n", abc_norm, flush=True)

print("ALL TESTS PASSED!", flush=True)
