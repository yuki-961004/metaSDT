"""Conversion helpers shared by Python frontend wrappers."""

import pandas


def extract_3d_mat(mat):
    if isinstance(mat, pandas.DataFrame):
        return [mat.values.tolist()]
    if isinstance(mat, dict):
        if "freq_mat" in mat:
            return mat["freq_mat"]
        if "prob_mat" in mat:
            return mat["prob_mat"]
        return [frame.values.tolist() for frame in mat.values()]
    return mat


def to_backend_df_dict(df):
    out = {}
    for col in df.columns:
        if (
            pandas.api.types.is_numeric_dtype(df[col])
            or pandas.api.types.is_bool_dtype(df[col])
        ):
            out[str(col)] = df[col].astype(float).tolist()
        else:
            codes, _ = pandas.factorize(df[col])
            out[str(col)] = codes.astype(float).tolist()
    return out


def coerce_fit(fit):
    if isinstance(fit, list):
        return pandas.DataFrame(fit)
    if isinstance(fit, dict):
        return {
            key: pandas.DataFrame(value) if isinstance(value, list) else value
            for key, value in fit.items()
        }
    return fit


def normalize_condition_keys(value, prefix="condition"):
    if not isinstance(value, dict):
        return value

    out = {}
    for index, (key, item) in enumerate(value.items()):
        normalized_key = str(key) if key is not None else ""
        normalized_key = normalized_key.strip()
        if normalized_key == "":
            normalized_key = f"{prefix}_{index}"
        out[normalized_key] = item
    return out


def build_estimator_block(estimator_name, control):
    control_used = dict(control) if isinstance(control, dict) else {}
    return {
        "name": estimator_name,
        "backend": "nlopt",
        "global_algorithm": control_used.get("algorithm"),
        "local_algorithm": control_used.get("local_algorithm"),
        "control": control_used,
    }


def as_meta_result(result, estimator_name, control):
    expected_keys = {"fit", "estimator", "diagnostics"}
    if isinstance(result, dict) and expected_keys.issubset(result.keys()):
        out = dict(result)
        out["fit"] = coerce_fit(out["fit"])
        if isinstance(out["fit"], dict):
            out["fit"] = normalize_condition_keys(out["fit"])
            if isinstance(out["diagnostics"], dict):
                out["diagnostics"] = normalize_condition_keys(
                    out["diagnostics"]
                )
        return out

    fit = coerce_fit(result)
    diagnostics = {}
    if isinstance(fit, dict):
        fit = normalize_condition_keys(fit)
        diagnostics = {key: {} for key in fit.keys()}

    return {
        "fit": fit,
        "estimator": build_estimator_block(estimator_name, control),
        "diagnostics": diagnostics,
    }
