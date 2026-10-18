# Candidate Brief

Refine the incumbent: `rule_gated_decade_familiarity` — rank 0, the best model on this data, ELPD-LOO -3158.4. Its hypothesis, as its author stated it:

> A refinement of `shared_digit_rule_preempts_parity` (final-digit similarity averaged over the examples, an abstracted "ends in that digit" rule pre-empting parity credit, a common-factor boost and decade-band familiarity all kept). The one change is that magnitude familiarity is a fallback: when people detect an explicit rule linking the examples — they all end in the same digit, or are all multiples of a common factor of 3 or more — the rule pre-empts "numbers near these", and the decade-band familiarity boost is cut by an inferred share; when the set has no detectable rule (a lone example, or examples with no shared digit or factor), people lean on magnitude neighbourhood at full strength. This addresses the critiques that people accept same-decade, different-digit targets after a single example more than predicted, and that generalization breadth differs between sets more than the model produces.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/rule_gated_decade_familiarity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
