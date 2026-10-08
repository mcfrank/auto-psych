# Candidate Brief

Refine the incumbent: `bayesian_chance_vs_repeating_motif_2` — rank 0, the best model on this data, ELPD-LOO -1003.0. Its hypothesis, as its author stated it:

> Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent a fair coin (which they believe over-alternates) explains it better than a "regular" generator, with the regular generators' unknowns averaged out normatively. The single change is one more regular generator in their hypothesis space: a "repeating motif" process that writes a short pattern (one to four flips long, e.g. H, HT, HHT, HHTT) and keeps copying it with an occasional slip (a fitted slip rate), so sequences that are near-repetitions of a short motif — perfect alternation and period-3/4 patterns above all — are explained as regular and look less random, addressing the critique that the incumbent under-penalises perfect alternation and periodic motifs.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/bayesian_chance_vs_repeating_motif_2.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
