# Candidate Brief

Refine the incumbent: `outlier_tolerant_rule_concepts` — rank 0, the best model on this data, ELPD-LOO -2796.5. Its hypothesis, as its author stated it:

> Refinement of `evidence_scarce_guessing_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100 (evidence weight growing with set size, guessing that becomes rarer as more examples are shown), but how readily they accept that an example could be a stray number the program produced outside its rule is their own inferred belief, rather than a fixed one-in-ten chance. The one change is this inferred tolerance for exceptions, which matters most for sets where all examples but one share a rule (e.g. 92, 82, 27, 42 or 71, 61, 26, 81): a tolerant learner keeps the near-fitting rule and says "yes" to far targets that fit it (52, 62; 11, 41), while an intolerant learner abandons the rule for a range spanning every example and rejects them.

Its source is `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/outlier_tolerant_rule_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
