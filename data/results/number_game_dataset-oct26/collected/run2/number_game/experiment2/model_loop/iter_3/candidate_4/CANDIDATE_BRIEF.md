# Candidate Brief

Refine the incumbent: `span_filling_size_exemplar` — rank 0, the best model on this data, ELPD-LOO -3105.9. Its hypothesis, as its author stated it:

> Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples on final digit, parity and graded shared factor (with coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, and still relax their criterion over the session. The single change is how closeness in size is judged: people treat the examples as marking out a filled-in stretch of the number line, so a target's size mismatch is how far it falls outside the span from the smallest to the largest example (zero anywhere inside it), not its distance from each example. The disagreement with the best model lives inside widely spaced sets — after 10, 90 the target 50 is as close in size as 10 itself and is endorsed far more often, so widely spread sets are generalized much more broadly than tight ones (the too-compressed between-set yes-rate spread in the critique).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/span_filling_size_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
