# Candidate Brief

Refine the incumbent: `kind_uncertainty_shifted_criterion` — rank 0, the best model on this data, ELPD-LOO -3663.5. Its hypothesis, as its author stated it:

> Refinement of `uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are, but the one change is what that uncertainty is about. It is not their spread of belief over individual concepts (which rule, which range width) but their uncertainty about what kind of program they face — a parity rule, a prime rule, a power rule, a multiples rule, a digit rule, or a magnitude range — so examples that leave it open whether the program follows an arithmetic rule or a stretch of the number line move the answer toward "no" (or "yes"), while examples that clearly point to one kind do not, however many concepts of that kind remain open.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/kind_uncertainty_shifted_criterion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
