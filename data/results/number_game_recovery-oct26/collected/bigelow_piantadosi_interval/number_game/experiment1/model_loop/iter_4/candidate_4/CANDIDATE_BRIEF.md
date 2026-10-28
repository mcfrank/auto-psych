# Candidate Brief

Refine the incumbent: `criterion_shifted_tempered_decision` — rank 0, the best model on this data, ELPD-LOO -778.8. Its hypothesis, as its author stated it:

> Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, examples' evidence tempered by an inferred weight) and pass their belief through the same inferred decision gain on its log-odds, but their yes/no criterion is not centred on even odds. The one change, taken from response_bias_concept_learning, is an inferred response bias that shifts the decision criterion by a constant on the log-odds scale, so the gain decides how sharply people separate strong from weak beliefs while the bias decides whether they lean toward "yes" or "no" — e.g. a liberal criterion gives more yeses to targets with middling or weak belief without making yeses to the examples themselves less likely.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/criterion_shifted_tempered_decision.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
