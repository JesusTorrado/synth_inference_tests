Synthetic tests for approximate inference frameworks
====================================================

Collection of synthetic (i.e. mock) tests from different sources for benchmarking approximate inference frameworks.

It contains scripts that can retrieve tests with different characteristics (see [tags]). Each test consists of a list of parameter bounds, a log-posterior (or simply log-probability) function, and high-precision MC samples from those posteriors tests.

Some tests are fixed, and others are randomised.

It can generate benchmark reports when fed some data manually.
