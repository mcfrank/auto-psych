# Candidate Brief

Refine the incumbent: `costly_squares_final_digit_concepts` — rank 0, the best model on this data, ELPD-LOO -2347.1. Its hypothesis, as its author stated it:

> Refinement of `final_digit_multiplicative_concepts`. People still build number concepts from mental primitives: a fitted cost per extra primitive, a fitted prior discount for salient "ends in d" rules, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept. The one change splits the single cost on repeated-multiplication concepts in two. "Perfect squares" (a number multiplied by itself) pays its own fitted extra cost, separate from the cost shared by powers of a base (and their exception variants) and cubes. The claim is that self-multiplication is the hardest of these to conceive, harder than repeatedly multiplying by a fixed base. So after square sets such as 100, 1, 16 or 4, 64, 25, 81, people rarely commit to "squares". Instead they fall back on other rules, ranges or scattered guesses. Meanwhile, powers of 2 or 3 stay more available than a single cost on all of these concepts allows.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/costly_squares_final_digit_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
