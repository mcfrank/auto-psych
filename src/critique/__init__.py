"""CriticAL-style posterior-predictive model criticism for PyMC cognitive models.

The :mod:`src.critique.ppc` module compares an incumbent fitted PyMC model
against the observed responses by computing LLM-proposed *test statistics* on
the observed data and on posterior-predictive replicates of the model, then
reporting the statistics whose observed value is a significant discrepancy
(two-sided empirical p-value, Benjamini–Hochberg FDR adjusted).

The approach follows CriticAL (Li et al., 2024, arXiv:2411.06590).
"""
