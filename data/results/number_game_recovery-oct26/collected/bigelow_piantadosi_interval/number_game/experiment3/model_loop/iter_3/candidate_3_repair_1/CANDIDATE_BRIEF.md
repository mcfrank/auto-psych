# Candidate Brief

Refine the incumbent: `serial_position_ends_inflated_size` — rank 0, the best model on this data, ELPD-LOO -3417.4. Its hypothesis, as its author stated it:

> Refinement of inflated_size_familiar_rule_prior. People do the same Bayesian concept learning as that model: 42 rules, with a familiarity prior that rises with rule size, and all ranges of 1-100. Perceived concept sizes are inflated by a constant, evidence is tempered separately for rules and ranges, the first example counts as a stronger or weaker draw, and answers pass through a decision gain and a shifted criterion. The one change adds a recency effect alongside that primacy effect: in sets of two or more, the last example shown is still fresh in memory when the target is judged, so it too counts as a stronger (or weaker) draw from the concept, by its own inferred multiplier. Concepts that contain both the first and the last example therefore gain belief over those that fit only the middle examples, and a multiplier of one recovers the incumbent.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/serial_position_ends_inflated_size.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
