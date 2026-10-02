# Candidate Brief

Refine the incumbent: `run_length_variety_preference` — rank 0, the best model on this data, ELPD-LOO -2272.9. Its hypothesis, as its author stated it:

> People judge randomness partly by the variety of a sequence's runs: a sequence whose runs of identical flips all have the same length (HTHTHTHT, HHTTHHTT, HHHTTTHH) has a regular, rhythmic run structure that looks designed, while one mixing runs of different lengths (HTTHHHTH) looks irregular and so random, by a shared weight on the number of distinct run lengths (relative to the most its length allows). This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity to it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and on mirror symmetry, and a personal guessing rate) and addresses the critique (surviving FDR) that among pairs matched on alternation rate people choose the sequence with more distinct run lengths more often than the best model predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/run_length_variety_preference.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
