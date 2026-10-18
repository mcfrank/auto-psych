# Candidate Brief

Refine the incumbent: `final_digit_rule_suppresses_parity` — rank 0, the best model on this data, ELPD-LOO -3169.2. Its hypothesis, as its author stated it:

> A refinement of `decade_band_digit_common_factor` (final-digit similarity, common-factor boost, decade-band familiarity all kept). The one change: the partial credit people give a target for merely sharing the examples' parity is not fixed but shrinks with how consistently the examples share one final digit — when the examples all end in the same digit, people infer an "ends in that digit" rule and give little or no credit to same-parity targets ending in a different digit, while for examples with varied final digits parity still earns its usual partial credit. This addresses the critique that, in sets whose examples share a final digit, people accept parity-only matches far less than the model predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_rule_suppresses_parity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
