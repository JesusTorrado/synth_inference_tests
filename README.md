Synthetic tests for approximate inference frameworks
====================================================

Collection of synthetic (i.e. mock) tests from different sources for benchmarking approximate inference frameworks.

It contains scripts that can retrieve tests with different characteristics (see [tags]). Each test consists of a list of parameter bounds, a log-posterior (or simply log-probability) function, and high-precision MC samples from those posteriors tests.

Some tests are fixed, and others are randomised.

It can generate benchmark reports when fed some data manually.

Conventions
-----------

PDF's are *Likelihoods*, and their `logpdf()` methods return log-likelihoods. They are defined on uniform priors given by their attribute `bounds`, if present, so that their log-posteriors are `log p(x) = logpdf(x) - sum_i(bounds_i[1] - bounds_i[0])`, defined by method `logpost()`.

`logpdf()` and similar methods take a vector of parameter values as a single argument, and are vectorized so that they can an array of such vectors.

**Note about implementing new PDF's**: The `logpdf()` method should not be overridden, since it includes wrapping for timing and evaluations count. For new PDF classes, implement the log-likelihood as `logp` instead, which will be called by `logpdf()`. `logp()` methods are expected to be vertorized as explained above.

Surrogate posterior or likelihood models from approximate inference frameworks are expected to work the same (or be wrapped such as they do) when using internally in this library (e.g. when passed to the `run.py` routines).
