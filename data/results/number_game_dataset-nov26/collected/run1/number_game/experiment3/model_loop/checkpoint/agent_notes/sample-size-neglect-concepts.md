---
name: sample-size-neglect-concepts
description: exp3 iter2 c1 — log-odds Bayes + set log-lik × n^-rho (sample-size neglect), MLE NLL 4249 vs base Bayes 4290; incumbent in-sample ~4100
metadata:
  type: project
---
experiment3 iter_2 candidate_1 (2026-11-02), `sample_size_neglect_concepts`: built on [[log-odds-distorted-concept-learning]] code; distortion = total set log-likelihood × n^(-rho), rho≈0.68 at MLE. Scipy MLE NLL on exp3 data: base log-odds Bayes 4289.9; +sample-size neglect 4249.4; +constant likelihood tempering 4264.3; +posterior power 4265.5. The Bayes family is still ~150 nats behind the exemplar incumbent (coincidence_decisive_exemplar, ELPD -4140).
**Why:** the brief asked for normative + one distortion. The critique was incoherent-set polarization, and none of these variants targets it directly.
**How to apply:** if Bayes slots recur, sample-size neglect beats tempering by about 15 nats. See [[frequency-weighted-multiple-concepts]].
