# Candidate Brief

Refine the incumbent: `zigzag_breadth_sequence_exemplar` — rank 0, the best model on this data, ELPD-LOO -4721.7. Its hypothesis, as its author stated it:

> Refinement of sequence_momentum_span_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, a relaxing criterion, the filled-in span and the momentum of the sequence's direction of travel). The single change is that people read how orderly the sequence of examples is in the order shown: when the examples march steadily in one direction (20, 40, 60) they take the program to be systematic and generalize narrowly in size beyond the examples, whereas when the sequence zigzags back and forth (40, 20, 60) they take the program to be scattering numbers widely and generalize broadly beyond them. The same numbers shown in a different order are thus generalized differently, and sets differ more in their breadth than a span-only model predicts (the critique's too-compressed between-set yes-rate spread).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/zigzag_breadth_sequence_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
