# metaSDT

`metaSDT` is a general modeling toolkit for Signal Detection Theory (SDT) models.
It uses C++ as the computational backend, with R and Python as frontends for data
analysis workflows.

The current implementation supports the basic SDT model. The package is designed
around a decoupled architecture so that new SDT-family models can be added
incrementally without rewriting the estimation stack. In general, adding a new
model should only require introducing a corresponding `model_<name>.cpp` module
that defines the model-specific probability structure.

## Estimation Methods

| Estimator | Backend |
| --- | --- |
| MLE | [NLopt](https://github.com/stevengj/nlopt) |
| MAP | [NLopt](https://github.com/stevengj/nlopt) |
| MCMC | [Stan Math](https://github.com/stan-dev/math) |
| ABC | [abcpp](https://github.com/yuki-961004/abcpp) |

## Roadmap

The long-term goal is to provide a unified framework for commonly used SDT and
meta-d' models, including the models summarized in Shekhar & Rahnev (2024)
(<https://doi.org/10.1037/xge0001524>). As the model layer grows, the existing
C++ backend and R/Python frontends will continue to share the same estimation
interfaces.
