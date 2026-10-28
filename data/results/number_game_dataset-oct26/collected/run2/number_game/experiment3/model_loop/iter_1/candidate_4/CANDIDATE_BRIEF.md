# Candidate Brief

Refine the incumbent: `enumerated_list_concept_exemplar` — rank 0, the best model on this data, ELPD-LOO -4719.5. Its hypothesis, as its author stated it:

> Refinement of zigzag_breadth_sequence_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, the relaxing criterion, the filled-in span, the momentum of the sequence's direction of travel and the zigzag-set breadth). The single change adds one concept to the space people consider: that the program simply produces a fixed list of particular numbers, of which the shown examples are members. This concept is consulted as separate evidence at the decision, and it applies only to targets that are themselves among the shown examples. People therefore say "yes" to a repeated example far more often than to a number only 1-3 away, even though the two are almost equally similar in size (the critique's exact-example boost).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/enumerated_list_concept_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
