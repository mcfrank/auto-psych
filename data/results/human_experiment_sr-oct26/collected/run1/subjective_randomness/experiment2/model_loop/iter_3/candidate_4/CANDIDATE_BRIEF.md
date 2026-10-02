# Candidate Brief

Refine the incumbent: `individual_alternation_sensitivity_mirror` — rank 0, the best model on this data, ELPD-LOO -2301.4. Its hypothesis, as its author stated it:

> Refinement of the incumbent `mirror_symmetry_streak_aversion_lapse`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry (palindromes look designed) by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that how sharply a departure from one's ideal switching rate counts against a sequence is each person's own trait (drawn from a population) rather than shared: some people are strict, consistent judges of alternation and others barely discriminate — addressing the critique that people differ in how consistently they agree with the majority choice more than the incumbent's person-level weights and lapse rates produce.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/individual_alternation_sensitivity_mirror.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
