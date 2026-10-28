# Candidate Brief

Refine the incumbent: `tempered_evidence_concept_guessing` — rank 0, the best model on this data, ELPD-LOO -805.0. Its hypothesis, as its author stated it:

> Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with the same occasional guessing, but they do not weigh the examples as fully independent random draws from the concept — each example's evidence counts for an inferred fraction (or multiple) of a full draw. The one change is this tempered size principle: weighting the evidence below one keeps broad rules and wide intervals plausible and widens generalization, while weighting it above one makes people lock onto the tightest concept consistent with the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/tempered_evidence_concept_guessing.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
