# Candidate Brief

Refine the incumbent: `length_normalised_chance_vs_motif` — rank 0, the best model on this data, ELPD-LOO -995.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is in how that evidence is weighed: people judge the evidence per flip rather than in total, so the log evidence difference is divided by a fitted power of the sequence length — the same per-flip regularity is judged about as decisively in a short pair as in a long one, instead of long sequences automatically yielding near-certain choices — addressing the critique that people are more decisive on short pairs (relative to long) than the incumbent predicts and that it over-penalises long perfect alternation.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/length_normalised_chance_vs_motif.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
