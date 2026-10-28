# Candidate Brief

Refine the incumbent: `top_marking_range_sampling` — rank 0, the best model on this data, ELPD-LOO -2059.5. Its hypothesis, as its author stated it:

> Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe the program shows where its range stops at the top — but they reason about this generatively rather than granting a fixed bonus. The one change: people assume that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random, so the largest example is explained by marking instead of by chance; a range ending at the largest example is therefore credited in proportion to how unlikely a random draw would have been to hit that number — far more for wide ranges than narrow ones, and less as more examples are shown — so single examples and sparse sets license especially strong downward generalization (after 60, "yes" to 40 but not 62), while dense sets fall back toward ordinary size-principle ranges.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/top_marking_range_sampling.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
