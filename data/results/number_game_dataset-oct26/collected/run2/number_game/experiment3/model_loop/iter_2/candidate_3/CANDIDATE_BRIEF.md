# Candidate Brief

Refine the incumbent: `saturating_criterion_relaxation` — rank 0, the best model on this data, ELPD-LOO -4712.9. Its hypothesis, as its author stated it:

> Refinement of enumerated_list_concept_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum and zigzag breadth). The single change is the time course of the relaxing yes-criterion: people loosen their criterion quickly over the first stretch of the session as they settle into a task with no feedback, and then it levels off at a stable, more lenient setting rather than continuing to relax steadily to the end. So yes rates rise sharply early in the session and then plateau, making mid-session answers about as lenient as late ones (the critique's mid-session plateau, which a steady linear drift cannot produce).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/saturating_criterion_relaxation.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
