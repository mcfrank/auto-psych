# Refinement menu

The models you may refine, other than the incumbent `ideal_alternation_with_balance`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### windowed_glimpse_expectation — rank 1, 0.1 ± 2.9 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1008.6

**Hypothesis:** People read a coin-flip sequence through a limited working-memory window, holding only the last few flips at a time, and compare how mixed each glimpse is (its balance of heads and tails) with how mixed they expect a glimpse of a random coin to be; a sequence looks random to the extent its glimpses match that expectation. The expectation comes from each person's own belief about how often a random coin switches sides, so some people expect near-perfect mixing and others streakier glimpses, and because the window spans more than two flips, lopsided heads/tails counts and long runs are judged beyond what the alternation rate alone shows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/windowed_glimpse_expectation.py`

### personal_ideal_alternation — rank 2, 0.4 ± 2.0 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -1008.9

**Hypothesis:** Each person carries their own ideal switching rate for a random coin (how often consecutive flips should differ), and judges a sequence as more random the closer its proportion of alternations lies to that personal ideal; these ideals differ between people, some expecting near-perfect alternation and others streakier sequences. The model disagrees most with the current best model on highly alternating sequences (e.g. HTHTHTHT versus a streaky sequence), where it predicts a population split — some people strongly prefer them, others strongly reject them — and on pairs that differ in balance but not in alternation rate, where it predicts indifference.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/personal_ideal_alternation.py`

### ideal_alternation_streak_penalty — rank 3, 0.9 ± 1.8 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -1009.4

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also notice the single longest streak of identical flips and treat a long streak as a sign of non-randomness, beyond what the overall alternation rate shows. The one change is a shared penalty on the length of the longest run (in flips), added because the critique shows that among irregular sequences people penalise long runs beyond what alternation rate captures, which the alternation-only model under-predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/ideal_alternation_streak_penalty.py`

### iter1_candidate1 — rank 4, 4.9 ± 10.2 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -1013.4

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent it is better explained by a fair coin than by the non-random generators one might suspect — a biased coin (which explains unbalanced heads/tails counts) or a sticky/switchy coin (which explains long runs or rigid alternation). The one distortion is that each person's mental model of a "fair" coin switches between heads and tails at their own personal rate rather than exactly half the time, so people disagree about alternation while all still penalise imbalance and runs that the alternative generators explain.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/iter1_candidate1.py`

### motif_stack_alternation_individual — rank 5, 35.3 ± 18.9 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1043.8

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the log-likelihood ratio of a fair coin versus Griffiths et al.'s four-motif stack automaton (the automaton held at the values the motif_stack fit settled on for this data), but people differ in how much they additionally favour or disfavour alternation — each participant carries their own preference for the sequence that switches between H and T more often, drawn from a population distribution. The single change is this participant-level alternation-preference random effect, added because the critique shows the between-participant spread in choosing the more-alternating sequence (SD 0.22) is four times what a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/motif_stack_alternation_individual.py`

### individual_alternation_prototype — rank 6, 40.6 ± 10.6 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1049.1

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular, i.e. close to a prototype alternation rate and not periodic). The one change: each person holds their own prototype alternation rate — some expect a random sequence to switch much more often than a fair coin does, others barely more or even less — drawn from a population distribution, instead of everyone sharing one prototype. This addresses the critique that people differ far more in how strongly they prefer the more-alternating sequence than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/individual_alternation_prototype.py`

### run_hazard_switch_belief — rank 7, 44.1 ± 19.2 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1052.6

**Hypothesis:** Refinement of `personal_switch_belief` (each person judges as more random the sequence that is more probable under their own subjective model of a coin as a process that switches between heads and tails with a personal probability). The one change: the believed coin has the gambler's fallacy built in — its chance of switching grows with the length of the current run (a shared rate of growth), on top of each person's own baseline switch belief — so a long run is judged far less probable than its number of switches alone implies. This addresses the critique that people penalise long runs beyond what the alternation rate captures.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/run_hazard_switch_belief.py`

### personal_ideal_longest_run — rank 8, 56.0 ± 27.6 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1064.5

**Hypothesis:** People judge randomness by the longest streak a sequence contains: each person carries their own ideal length for the longest run of identical flips in a random sequence (relative to its length), and picks the sequence whose longest streak is closer to that personal ideal. Some people expect random sequences to contain almost no streaks, others expect a noticeable streak, so the same pair can be judged in opposite directions; the rest of the sequence's structure (its overall alternation rate or H/T balance) plays no role beyond what it does to the longest streak.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/personal_ideal_longest_run.py`

### individual_irregularity_weight — rank 9, 71.4 ± 20.2 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1079.9

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/individual_irregularity_weight.py`

### personal_switch_belief — rank 10, 82.1 ± 21.4 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1090.6

**Hypothesis:** Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/personal_switch_belief.py`

### local_representativeness — rank 11, 390.0 ± 37.3 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1398.4

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/local_representativeness.py`

### motif_stack — rank 12, 408.1 ± 40.3 nats behind the best (10.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1416.5

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### online_transition_surprise — rank 13, 547.3 ± 52.6 nats behind the best (10.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1555.7

**Hypothesis:** People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/online_transition_surprise.py`

### falk_konold_dp — rank 14, 570.9 ± 44.4 nats behind the best (12.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1579.4

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### finite_experience_occurrence — rank 15, 739.8 ± 68.3 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.3

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
