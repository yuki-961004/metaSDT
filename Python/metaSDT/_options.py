"""Default option helpers for Python shell functions."""

import warnings


def default_shell_run_m_option():
    """Return default shell-level options for :func:`shell_run_m`."""
    return {
        "n": 1000,
        "seed": None,
        "plot": True,
        "density_points": 512,
        "xlim": None,
        "show": True,
    }


def merge_shell_run_m_option(option=None):
    if option is None:
        option = {}
    if not isinstance(option, dict):
        raise TypeError("option must be a dict or None.")

    merged_option = default_shell_run_m_option()
    unknown_keys = sorted(set(option.keys()) - set(merged_option.keys()))
    if unknown_keys:
        warnings.warn(
            "Unknown shell_run_m option field(s): "
            + ", ".join(str(key) for key in unknown_keys),
            RuntimeWarning,
            stacklevel=2,
        )
    merged_option.update(option)
    return merged_option
