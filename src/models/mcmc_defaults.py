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

# Convergence gate (admission, export, pruning): a fit is converged when at
# most MAX_DIVERGENCE_FRACTION of its transitions diverged and R-hat <= MAX_R_HAT
# and bulk ESS >= MIN_BULK_ESS on every free parameter. User decision
# 2026-09-26: the strict gate of Vehtari et al. (2021) — R-hat 1.01, ESS 400, no
# divergences — rejected most seed models at the sweep's target_accept.
MAX_R_HAT = 1.05
MIN_BULK_ESS = 100
MAX_DIVERGENCE_FRACTION = 0.001
# A fit that fails the gate but is a near miss (below) is refit once at this
# target_accept (smaller NUTS steps), and that fit is the model's fit from then
# on.
ESCALATED_TARGET_ACCEPT = 0.95

# A near miss: a failed fit that smaller steps can plausibly fix — at most
# NEAR_MISS_MAX_DIVERGENCE_FRACTION of its transitions diverged, max R-hat <=
# NEAR_MISS_MAX_R_HAT and bulk ESS >= NEAR_MISS_MIN_BULK_ESS on every free
# parameter. Anything worse (a chain stuck in another mode, R-hat ~1.5-2.5 and
# bulk ESS ~5; a quarter of all transitions divergent) is a geometry problem
# that the refit only repeats, at twice the cost: in the September 2026 sweep
# such refits of variants of one slow seed model took ~40 min each before the
# same rejection. The failures in the sweep's ledgers were bimodal — 0.3-0.7%
# divergent with R-hat and ESS fine, or 25% divergent with R-hat >= 1.5 and
# ESS <= 7 — and these thresholds sit in the gap between the two modes. A fit
# with no divergence statistic (not NUTS) is never a near miss.
NEAR_MISS_MAX_R_HAT = 1.2
NEAR_MISS_MIN_BULK_ESS = 20
NEAR_MISS_MAX_DIVERGENCE_FRACTION = 0.02

# Wall-clock limit on each sampling run of a candidate's admission fit (the
# first fit and a near-miss refit are limited separately). A candidate still
# sampling at the limit is stopped (its process group is killed) and rejected
# as too slow to fit. Seeds and carried models are never time-limited.
CANDIDATE_FIT_TIME_LIMIT_SEC = 15 * 60

# The candidate agent's self-check (src/pipelines/inner_loop/check_candidate.py):
# a smoke fit that proves the model loads, samples and scores — never a
# production fit. One chain on one core keeps it cheap inside an agent
# session; production sampling happens at admission with the values above.
CANDIDATE_CHECK_DRAWS = 100
CANDIDATE_CHECK_TUNE = 100
CANDIDATE_CHECK_CHAINS = 1
