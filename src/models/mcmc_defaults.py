"""Single source of truth for MCMC sampler settings.

Every entry point (outer-loop run CLI, inner-loop run CLI, the PPC critique
CLI, design-time family-twin fits) reads its defaults from here, so "what
settings did this run use?" has one answer unless a config overrides it
explicitly. Before this module the defaults had drifted per entry point
(outer 2000/2000/4, inner CLI 500/500/2, ppc 2000/2000/4, design twins
hard-coded 500/500/2).
"""

# Production sampling: full posteriors for model comparison (ELPD-LOO needs
# reliable log-likelihood draws). Quick local runs should pass explicit lower
# values rather than lowering these. draws/tune raised and target_accept lifted
# off PyMC's implicit 0.8 default to shrink divergences and stabilise the
# PSIS-LOO tail — high Pareto-k on the best-fitting model was hard-failing the
# inner loop's export gate.
PRODUCTION_DRAWS = 4000
PRODUCTION_TUNE = 3000
PRODUCTION_CHAINS = 4
PRODUCTION_TARGET_ACCEPT = 0.99

# Design-time fits (posterior-informed exhaustive design, experiments >= 2):
# cheaper on purpose — the design step only needs posterior-predictive means
# to weight EIG scenarios, not publication-grade posteriors.
DESIGN_TWIN_DRAWS = 500
DESIGN_TWIN_TUNE = 500
DESIGN_TWIN_CHAINS = 2
