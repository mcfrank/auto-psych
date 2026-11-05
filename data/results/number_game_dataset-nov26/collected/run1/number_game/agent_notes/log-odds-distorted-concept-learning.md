---
name: log-odds-distorted-concept-learning
description: exp2 iter0 c0 — Bayesian rules+intervals with log-odds-linear probability distortion; MLE NLL 2800 vs pure Bayes 3150
metadata:
  type: project
---
experiment2 iter_0 candidate_0 (2026-11-02), `log_odds_distorted_concept_learning`: rules (parity, primes, squares, cubes, pow2-10, mult3-50, end d) + intervals (Erlang width prior), size principle with noise; p_yes = sigmoid(bias + gamma*logit(posterior predictive)). Scipy MLE: pure Bayes 3150, + distortion 2826 (gamma≈0.4, noise≈0.3), + digit-vs-math rule-family split 2800 (digit share ≈0.5). Still likely ~60-80 nats behind exemplar incumbent in-sample.
**Why:** brief asked normative + one distortion; aimed at gcd-multiples critique.
**How to apply:** Bayesian accounts' main problem was overconfident posterior→response mapping; see [[mixture-variant-mle-screen]].
