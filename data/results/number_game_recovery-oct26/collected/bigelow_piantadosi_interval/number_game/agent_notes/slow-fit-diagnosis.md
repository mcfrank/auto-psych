---
name: slow-fit-diagnosis
description: How to diagnose a too-slow candidate fit (Hessian at mode, leapfrog steps/iter) and what fixed schema_grammar_rule_prior
metadata:
  type: project
---
On 2026-10-21 (experiment3 iter_2 candidate_2_repair_1), schema_grammar_rule_prior timed out. At its mode the 8 Dirichlet schema weights, p_mix and size_exponent were constrained only by their priors: Hessian eigenvalues of about 2 matched the Dirichlet(2) prior. The fit was degenerate (gain about 0.14, rule evidence weight about 4, log-lik -4454 against the incumbent's -3190).

The repair dropped the schema weights and size_exponent (a fixed uniform grammar prior), giving rule_grammar_uniform_prior. That brought the fit to about 15 leapfrog steps per iteration (incumbent about 28), with no divergences.

A centre-weighted range sampling idea (shape (4u(1-u))^peak) was also tried: its MAP peak was about 0.06 and it gained only 0.3 nats, so it is a near-duplicate of the incumbent. Do not retry it.

**Why:** if parameters are pinned only by their priors, or the mode is degenerate, NUTS takes long trajectories.
**How to apply:**
- To diagnose, run L-BFGS on step._logp_dlogp_func, then look at the Hessian eigenvalues and the correlation matrix.
- To time a fit, call src.models.pymc_inference._warm_start_sample with tune=draws=40 and chains=1, then read sample_stats n_steps.

See [[volatile-model-speed]] and [[map-screening-refinements]].
