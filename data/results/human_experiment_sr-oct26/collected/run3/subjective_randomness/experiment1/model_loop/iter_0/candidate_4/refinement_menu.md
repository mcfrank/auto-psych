# Refinement menu

The models you may refine, other than the incumbent `local_representativeness`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### motif_stack — rank 1, 42.6 ± 14.9 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1302.5

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### falk_konold_dp — rank 2, 276.4 ± 31.9 nats behind the best (8.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1536.3

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### finite_experience_occurrence — rank 3, 477.7 ± 53.3 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1737.6

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
