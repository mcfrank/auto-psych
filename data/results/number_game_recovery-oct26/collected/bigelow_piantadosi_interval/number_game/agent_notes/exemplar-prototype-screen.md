---
name: exemplar-prototype-screen
description: iter_4 MAP screen of exemplar/prototype replacements for interval concepts — all lose 120-195 nats vs uneven_rule_familiarity_ends
metadata:
  type: project
---
On 2026-10-21 (experiment3 iter_4 candidate_1, exemplar/prototype brief), the base was uneven_rule_familiarity_ends with anchor/recency dropped (MAP loglik -3186.3). I replaced the 5,050 intervals with a magnitude mode. MAP log-lik for each version:
- Exemplar: similarity to the nearest shown example, with widths on a grid and a sequential kernel-generation likelihood. Exponential gradient -3381, Gaussian -3383, plus a width-prior exponent -3379. The loss is spread across all target categories; singleton near-targets are badly underpredicted.
- Prototype: a latent centre × width grid, examples iid from a kernel around it, target judged by similarity to the centre. Exponential -3323. Gaussian, widths 0.5-60: -3306.7 (best). Widths 1-120 gave -3329, widths 0.5-30 gave -3315.
I submitted the Gaussian prototype as magnitude_prototype_similarity. It is expected to lose by about 120 nats, a falsification result: crisp intervals beat similarity gradients.
**Why:** saves re-screening exemplar/prototype variants of the magnitude component.
**How to apply:** don't expect any similarity-gradient replacement of intervals to compete; see [[map-screening-refinements]].
