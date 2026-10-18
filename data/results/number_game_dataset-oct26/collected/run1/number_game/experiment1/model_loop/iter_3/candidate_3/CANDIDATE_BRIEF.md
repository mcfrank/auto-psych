# Candidate Brief

Refine the incumbent: `final_digit_similarity_gradient` — rank 0, the best model on this data, ELPD-LOO -1664.1. Its hypothesis, as its author stated it:

> People ignore magnitude and judge the target by how much its final digit resembles the final digits of the examples: a target ending in the same digit as an example is maximally similar to it, one ending in a different digit of the same parity (odd/even) is partly similar, and one of the opposite parity is least similar. They say "yes" in proportion to the target's average final-digit similarity to all the examples shown, so the answer is graded by the share of examples it resembles rather than by any rule that every example must obey, and numerically close targets get no credit for closeness.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/final_digit_similarity_gradient.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
