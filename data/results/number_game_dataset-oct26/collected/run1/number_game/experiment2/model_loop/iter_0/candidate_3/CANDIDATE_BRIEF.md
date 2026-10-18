# Candidate Brief

Refine the incumbent: `digit_similarity_with_common_factor` — rank 0, the best model on this data, ELPD-LOO -3207.7. Its hypothesis, as its author stated it:

> A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), but the one change is that they also notice the largest whole number (3 or more) that divides every example and treat "multiples of that number" as a candidate rule, so a target divisible by the examples' common factor gets an extra, inferred boost toward "yes" whatever its final digit. This addresses the critique that, among parity-matched targets, people accept those divisible by the examples' common divisor (e.g. multiples of 8 after 16, 8, 64) more than the final-digit model predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/digit_similarity_with_common_factor.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
