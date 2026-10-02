# Candidate Brief

Refine the incumbent: `personal_pattern_detection_gain` — rank 0, the best model on this data, ELPD-LOO -3451.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent `personal_complement_symmetry_aversion`: each sequence is still judged by its switching rate against a personal ideal (personal sensitivity, lenient toward over-alternation), personal weights on heads/tails imbalance and the longest streak, the structural cues of a designed sequence (the final streak, mirror symmetry, complement symmetry, run-length variety and the most lopsided stretch), and a personal guessing rate. The one change is that noticing structure is a single personal trait rather than a trait specific to antisymmetry: each person has their own pattern-detection gain, drawn from a population, that scales all the structural cues together (now shared in their relative weights), so a person who spots palindromes also spots antisymmetry, rhythmic runs and local excesses, while another sees none of them — predicting that people's agreement with the majority is a stable personal trait that varies more across people than cue-specific weights produce (the critique's excess spread and split-half stability of majority agreement).

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_pattern_detection_gain.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
