# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_mixture — rank 1, 112.8 ± 16.5 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
