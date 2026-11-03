# Candidate Brief

Refine the incumbent: `without_replacement_size_principle` — rank 0, the best model on this data, ELPD-LOO -1931.7. Its hypothesis, as its author stated it:

> Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (equal rule prior, 10% of examples random, occasional guessing at their own yes rate), but they understand that the program shows distinct examples, so each further example is read as a draw from the concept's members not yet shown (sampling without replacement) rather than from all its members again. The one change is this without-replacement size principle, which adds no free parameter and predicts that several examples sharpen generalization toward small concepts more than the standard model does: a few examples that nearly exhaust a small rule (e.g. 16, 8, 64 for powers of 2) or a short range make that concept far more credible than broader alternatives.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment2/model_loop/models/without_replacement_size_principle.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
