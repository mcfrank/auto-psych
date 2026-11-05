# Candidate Brief

Refine the incumbent: `multiple_route_exemplar` — rank 0, the best model on this data, ELPD-LOO -2721.3. Its hypothesis, as its author stated it:

> People judge a target by its summed similarity to the examples (closeness in magnitude, same final digit, same parity, and whether it divides by the set's common divisor), sharpened when the examples share a suspicious coincidence. On top of that there is a second, separate way for a target to resemble an example: being one of its multiples (14, 49 or 98 after "7"; 64 after "4"). This multiples route adds a fixed amount of similarity no matter how far apart the two numbers are, instead of only scaling down closeness. So multiples of an example that are far from it and share none of its digits are still often endorsed, especially after single examples and small-number examples, where magnitude similarity alone predicts almost certain rejection.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/multiple_route_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
