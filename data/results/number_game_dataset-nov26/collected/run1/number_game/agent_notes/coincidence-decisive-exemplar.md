---
name: coincidence-decisive-exemplar
description: exp3 iter1 c5: coincidence moved from similarity sharpening to decision slope on parity_salience, +23 nats MAP
metadata:
  type: project
---
exp3 iter1 candidate_5 `coincidence_decisive_exemplar`, a refinement of `parity_salience_asymmetry_exemplar`. Suspicious coincidence no longer scales log-similarity (the old `gain`). It scales the response slope instead: beta*exp(sharpening*coinc). Same number of parameters. MAP log-lik -4077.6 vs base -4100.4 (+23); on set_round_kinship_averaging the same swap gave +16.7. Aimed at the cell_polarization critique (q=0.018). Smoke log-lik -4085.2.
Screened on the same base with little gain: a set-size term on magnitude width (+0.1), set-size gain (+1.6), set-size response slope (+3.8), a free single-example width (+0.4). Setting single-example spread to 0 lost 10 nats.
**Why:** the coincidence effect acts on decisiveness rather than on generalization width, so later refinements should build on the slope form.
**How to apply:** coincidence-decisiveness may become saturated if the incumbent's own slots adopt it. Related: [[coincidence-sharpened-exemplar]], [[parity-salience-asymmetry-exemplar]].
