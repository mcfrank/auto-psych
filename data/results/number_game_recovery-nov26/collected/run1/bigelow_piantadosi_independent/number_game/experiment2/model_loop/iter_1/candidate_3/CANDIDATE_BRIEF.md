# Candidate Brief

Refine the incumbent: `final_digit_salient_rules` — rank 0, the best model on this data, ELPD-LOO -1773.7. Its hypothesis, as its author stated it:

> Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and moderate-width ranges sampled along a compressed number line, but they do not consider all rules equally plausible beforehand — rules that can be read straight off a numeral's final digit (odd, even, "ends in d") are perceptually salient and get a fitted prior boost over the other arithmetic and digit rules. The one change is this final-digit salience in the rule prior: when all examples share a final digit or a parity, people commit to that rule more firmly than to a range, so they endorse distant same-ending and same-parity numbers more and reject opposite-parity neighbours (one step from an example) more than the incumbent predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_salient_rules.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
