# Candidate Brief

Refine the incumbent: `round_landmark_veto` — rank 0, the best model on this data, ELPD-LOO -3485.4. Its hypothesis, as its author stated it:

> Round numbers (10, 20, ..., 100) are landmarks on the mental number line rather than ordinary outputs, so people check a round target against the examples before accepting it: if none of the examples is itself a round number, on a fitted share of such trials they doubt that the program produces round numbers and answer no, whatever concept they inferred. Concept inference itself is unchanged from the current best model, so the claim shows up only after sets with no multiple of 10 (e.g. 8, 64 or 32, 4, 16), where people say no to 10, 20, 30 or 70 more often than the concept they inferred (e.g. "even numbers") implies.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/round_landmark_veto.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
