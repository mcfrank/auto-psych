# Candidate Brief

Refine the incumbent: `rule_breadth_prior_concepts` — rank 0, the best model on this data, ELPD-LOO -907.8. Its hypothesis, as its author stated it:

> Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random), with occasional guessing at their own yes rate — but before seeing any examples they already prefer rules by their breadth, giving each rule a prior weight that rises or falls as a power of how many numbers it contains (a preference for specific, "interesting" rules over broad ones, or the reverse). The one change is replacing the flat prior across rules with this single size-graded rule prior, which predicts that broad rules such as "odd" or "even" win less (or more) of the posterior than an equal-prior learner gives them, e.g. fewer "yes" answers to arbitrary odd numbers after 85, 19, 91.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/rule_breadth_prior_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
