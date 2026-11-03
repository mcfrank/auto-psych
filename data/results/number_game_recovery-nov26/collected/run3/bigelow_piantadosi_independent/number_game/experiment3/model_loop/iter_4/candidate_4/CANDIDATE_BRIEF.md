# Candidate Brief

Refine the incumbent: `forward_continuation_response_bias` — rank 0, the best model on this data, ELPD-LOO -3483.0. Its hypothesis, as its author stated it:

> At the answer stage people treat the examples as what the program has produced so far, and they assume it runs upward. So, beyond what the concept they inferred implies, they lean toward yes for a target above the largest example and toward no for a target below the smallest. The strength of this lean is fitted, and targets inside the examples' span are unaffected. Concept inference is the current best model's, unchanged (including its round-number veto). So after 32, 4, 16, a target such as 40 or 60 gets more yeses than 2 or 3, even where the inferred concept rates them equally.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/forward_continuation_response_bias.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
