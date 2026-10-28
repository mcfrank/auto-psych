# Candidate Brief

Refine the incumbent: `sequence_momentum_span_exemplar` — rank 0, the best model on this data, ELPD-LOO -3100.8. Its hypothesis, as its author stated it:

> People read the examples as a sequence the program is producing in the order shown and expect it to keep going in the direction it has been moving: beyond the span the examples fill in, a number lying ahead of the last-shown example in the direction of travel from the first example to the last feels much closer in size than an equally distant number lying back behind where the sequence started. So after 20, 40 people generalize to 50-60 more than to 0-10, but after 40, 20 they generalize to 10 more than to 50 — a momentum in the order of presentation, not a fixed upward preference, which produces the critique's recency (closer-to-last) advantage.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/sequence_momentum_span_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
