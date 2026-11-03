---
name: two-source-familiar-glitches
description: exp3 iter_0 c5 repair: two_source with rule-count familiar glitches (g≈.26), +0.43 MAP, min pool RMSE .0025 (incumbent, bounded APs); exemplar+rule-count was a near-dup (.0016)
metadata:
  type: project
---
exp3 iter_0/candidate_5_repair_1 (2026-11-01) is two_source_familiar_glitches. It is two_source_program_concepts with glitches q = (1-g)/100 + g*rulecount(x)/sum. Second-rule tables are folded by j; the glitch tables keep the 16-subset axis, and the model multiplies in prod 100q over the strays. At g=0 it matches the original to 1e-14.
- MAP ll: -3322.25, against -3322.68 for the original two_source. MAP values: g .26, mix .038.
- Novelty-pool RMSE at MAP: .0025 vs the incumbent and vs bounded_arithmetic_progressions; .0028 vs familiar_strays_equal; .0042 vs two_source.
- Check OK.

Rejected first try: graded_exemplar_equal_rule_prior with rule-count familiarity. It gains +0.7 MAP, but its pool RMSE vs that parent is only .0016, a near-dup.

Exact novelty check: fit find_MAP on the responses, set_data(prepare_observed(novelty_pool.json rows with rating 0)), then evaluate pytensor.function(value_vars, replace_rvs_by_values([p_yes])). This reproduced admission's .00179 as .00184. RMSE on the training stimuli overstates it about 2x. Script: /tmp/c5r_pool.py (may be gone).

**How to apply:** before submitting, always compute the pool RMSE against the nearest models. See [[two-source-program-concepts]], [[two-source-shared-program-prior]], [[range-limited-rule-familiar-strays]].
