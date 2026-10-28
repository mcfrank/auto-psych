# Candidate Brief

Refine the incumbent: `first_example_evidence_weighting` — rank 0, the best model on this data, ELPD-LOO -1967.1. Its hypothesis, as its author stated it:

> Refinement of first_example_anchored_concepts: people still let the first example they see anchor their Bayesian concept learning over weighted rules and numeric ranges (with evidence tempered separately for rules and ranges, and a decision gain and shifted criterion), but the anchoring works through evidence rather than a flat bonus: the first example is weighed as a stronger (or weaker) draw from the concept than later examples, by an inferred multiplier on its size-principle evidence. The one change replaces the incumbent's equal boost to every concept containing the first example with this extra evidential weight, so the first example pulls belief most strongly toward small, specific concepts that contain it (e.g. "powers of 2" when 16 is shown first) and only weakly toward broad ones like "even numbers" or wide ranges, while concepts that must write it off as noise are penalised more.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/first_example_evidence_weighting.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
