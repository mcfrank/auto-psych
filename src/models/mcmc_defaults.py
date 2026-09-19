"""Single source of truth for MCMC sampler settings.

Every entry point (outer-loop run CLI, inner-loop run CLI, the PPC critique
CLI, design-time family-twin fits) reads its defaults from here, so "what
settings did this run use?" has one answer unless a config overrides it
explicitly.
"""

# Production sampling: full posteriors for model comparison (ELPD-LOO needs
# reliable log-likelihood draws). Quick local runs should pass explicit lower
# values rather than lowering these. A model whose posterior geometry makes
# 0.99 needlessly expensive can declare its own SAMPLER_SETTINGS in its file.
PRODUCTION_DRAWS = 4000
PRODUCTION_TUNE = 3000
PRODUCTION_CHAINS = 4
PRODUCTION_TARGET_ACCEPT = 0.99

# Chains run in parallel, one worker process each. Matches PRODUCTION_CHAINS
# so every chain gets its own core. Callers that fit many models concurrently
# should pass cores=1 explicitly to avoid oversubscribing the machine.
PRODUCTION_CORES = 4

# Design-time fits (posterior-informed exhaustive design, experiments >= 2):
# cheaper on purpose — the design step only needs posterior-predictive means
# to weight EIG scenarios, not publication-grade posteriors.
DESIGN_TWIN_DRAWS = 500
DESIGN_TWIN_TUNE = 500
DESIGN_TWIN_CHAINS = 2
