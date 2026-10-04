# Refinement menu

The models you may refine, other than the incumbent `local_representativeness`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### motif_stack — rank 1, 47.0 ± 21.6 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1427.4

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### falk_konold_dp — rank 2, 184.1 ± 28.5 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1564.5

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### finite_experience_occurrence — rank 3, 322.8 ± 47.6 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1703.2

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

### personal_markov_coin_belief — no comparison row

**Hypothesis:** Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_markov_coin_belief.py`

### personal_alternation_ideal — no comparison row

**Hypothesis:** Each person carries their own ideal switching rate for a random coin — some expect a fair coin to flip side about half the time, others expect it to alternate far more often — and judges a sequence as random to the extent that its proportion of H/T switches is close to that personal ideal. The current best model assumes one shared alternation prototype for everyone; this model disagrees most sharply on pairs where both sequences alternate heavily (e.g. HTHTHTHT versus HTHHTHTH), which strong over-alternators should split decisively toward the more alternating sequence while moderate people choose the other.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_alternation_ideal.py`

### person_alternation_prototype — no comparison row

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky local representativeness: a sequence looks random when it is locally balanced and irregular, irregularity being closeness to an over-alternating prototype plus a periodic-template penalty). The one change: each person carries their own ideal alternation rate (prototype), drawn from a population distribution, rather than everyone sharing a single prototype — so people differ in how much alternation they expect from a random coin, which the single-population incumbent cannot produce (the critique's large across-participant spread in preference for the more-alternating sequence).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/person_alternation_prototype.py`

### person_specific_alternation_prototype — no comparison row

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge randomness by local H/T balance across scales plus irregularity measured as distance from a preferred alternation rate and a periodic-template penalty, but each person holds their own preferred alternation rate (prototype) rather than one shared by everyone. The single change is making the alternation prototype person-specific, drawn from a population distribution, which addresses the critique that the model under-produces individual differences in preference for alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/person_specific_alternation_prototype.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
