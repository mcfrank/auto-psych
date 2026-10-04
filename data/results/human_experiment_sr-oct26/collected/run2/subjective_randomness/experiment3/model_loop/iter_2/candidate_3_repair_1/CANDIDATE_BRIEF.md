# Candidate Brief

Refine the incumbent: `person_pattern_sensitivity_rule_built` — rank 0, the best model on this data, ELPD-LOO -3285.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `rule_built_tally_switch_triplet` (running heads-minus-tails tally judged by closeness of its span to a personal expected span with person-specific, length-scaled sensitivity; switching judged by closeness to a shared ideal rate with person-specific weight; a reward for variety of three-flip chunks; a penalty on sequences built by a copying or mirror rule; a personal left/right lean). The one change: people differ in how much they attend to the pattern structure of a sequence — each person has their own pattern sensitivity, a single trait that scales both structural cues together (chunk variety and the rule-built penalty), so some people reliably reject recycled-chunk and rule-built sequences while others barely notice them and judge only by tally and switching. This addresses the critique (surviving FDR) that participants differ in how often they agree with the majority more than the incumbent allows, locating that heterogeneity in pattern perception rather than in decision noise or lapses.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/person_pattern_sensitivity_rule_built.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
