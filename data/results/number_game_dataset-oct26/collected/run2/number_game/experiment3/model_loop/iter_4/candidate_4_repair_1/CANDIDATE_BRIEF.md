# Candidate Brief

Refine the incumbent: `leading_digit_string_similarity` — rank 0, the best model on this data, ELPD-LOO -4673.1. Its hypothesis, as its author stated it:

> Refinement of upward_count_start_boundary (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth, the saturating criterion relaxation and the upward-count start boundary). The single change is that people also compare numbers as written digit strings by their leading digit: a target whose first written digit matches an example's (34 after 31 or 3, 72 after 79) feels more similar to that example, all-or-none, with attention to this match stretched when the examples themselves share a leading digit. So targets that share the examples' leading digit are endorsed more, and those that do not are endorsed less, than numerical distance alone predicts — even across decade-sized gaps between one- and two-digit numbers (3 and 35).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/leading_digit_string_similarity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
