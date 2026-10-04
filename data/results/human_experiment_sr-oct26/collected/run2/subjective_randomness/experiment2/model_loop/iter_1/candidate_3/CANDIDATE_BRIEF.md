# Candidate Brief

Refine the incumbent: `balance_aware_heads_alternation_ideal` — rank 0, the best model on this data, ELPD-LOO -2383.0. Its hypothesis, as its author stated it:

> Refinement of the incumbent `heads_default_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences) and still favour heads-leaning sequences, but they also expect a random coin to come out roughly half heads and half tails, so a sequence whose heads and tails counts are lopsided in either direction looks less random. The one change is a shared symmetric penalty on the H/T imbalance (|#H − #T| as a share of the length), addressing the critique that among pairs with equal switch counts people choose the more balanced sequence far more often than the incumbent's one-directional heads-share term allows.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/balance_aware_heads_alternation_ideal.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
