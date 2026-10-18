# Candidate Brief

Refine the incumbent: `digit_rule_size_principle_counting` — rank 0, the best model on this data, ELPD-LOO -4689.4. Its hypothesis, as its author stated it:

> A refinement of `counting_up_from_single_example` (final-digit similarity judged by its shortfall from the examples' mutual similarity, rule-pre-empted parity credit, a privileged round common-factor rule, rule-gated decade-band familiarity and step-costed counting up from single examples, all kept). The one change is a size principle on the "ends in that digit" rule: each further example beyond the second that shares the set's final digit is another suspicious coincidence that firms up belief in the rule, so a target that breaks the shared digit is rejected by an inferred, additional amount per confirming example (e.g. 37 is rejected more after 13, 43, 73, 93 than after 13, 43), whereas the incumbent's mutual-similarity shortfall is the same for two or four uniform examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/digit_rule_size_principle_counting.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
