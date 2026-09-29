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
# Design-time fits use a model's own SAMPLER_SETTINGS target_accept when it
# declares one, else this: a compromise between the production 0.99 (slow,
# tiny steps) and PyMC's 0.8 (user decision, 2026-09-26).
DESIGN_TWIN_TARGET_ACCEPT = 0.9

# The candidate agent's self-check (src/pipelines/inner_loop/check_candidate.py):
# a smoke fit that proves the model loads, samples and scores — never a
# production fit. One chain on one core keeps it cheap inside an agent
# session; production sampling happens at admission with the values above.
CANDIDATE_CHECK_DRAWS = 100
CANDIDATE_CHECK_TUNE = 100
CANDIDATE_CHECK_CHAINS = 1
