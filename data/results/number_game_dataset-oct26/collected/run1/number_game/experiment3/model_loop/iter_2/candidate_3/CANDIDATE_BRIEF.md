# Candidate Brief

Refine the incumbent: `round_counting_privileged_factor` — rank 0, the best model on this data, ELPD-LOO -4707.8. Its hypothesis, as its author stated it:

> A refinement of `shortfall_from_mutual_similarity` (final-digit similarity judged by its shortfall from the examples' mutual similarity, rule-pre-empted parity credit and rule-gated decade-band familiarity, all kept). The one change is that the "multiples of the examples' common factor" rule is not equally compelling for every factor: counting by fives and tens is a privileged, overlearned sequence, so when the examples' common factor is a multiple of 5 (e.g. 5, 65 or 90, 100, 5) people embrace "multiples of that factor" far more strongly than they do for factors like 3, 4, 6 or 8, with the two strengths inferred separately. The sharpest disagreement with the incumbent lies in round targets after round sets (e.g. 100, 80, 50 after 5, 65 accepted much more) and in multiples of non-round factors (e.g. 64 after 8, 80, 48 accepted much less).

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/round_counting_privileged_factor.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
