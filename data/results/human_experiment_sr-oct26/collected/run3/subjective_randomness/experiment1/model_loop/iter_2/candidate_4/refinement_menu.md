# Refinement menu

The models you may refine, other than the incumbent `bayesian_overalternating_chance_model`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### iter0_candidate3 — rank 1, 1.4 ± 12.2 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1023.0

**Hypothesis:** Refinement of `local_representativeness`: people judge randomness by the same Kahneman & Tversky local-representativeness score (multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates), but they differ in how decisively they apply it. The single change is that the decision sensitivity (beta) is person-specific, drawn from a population distribution, rather than shared by everyone — addressing the critique that people differ in agreement with the majority far more than one pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter0_candidate3.py`

### iter0_candidate4 — rank 2, 1.8 ± 12.1 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1023.4

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random to the extent it is locally balanced at several window scales and irregular, i.e. somewhat over-alternating but not periodic). The one change: people share this representativeness criterion but differ in how consistently they apply it, so each participant has their own decision sensitivity (drawn from a population distribution) rather than one shared sensitivity — addressing the critique that participants differ in agreement with the majority far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter0_candidate4.py`

### asymmetric_alternation_representativeness — rank 3, 2.3 ± 12.5 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -1024.0; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific sensitivity): people judge randomness by the same score — multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates — but the deviation from their prototype alternation rate is felt asymmetrically. The single change: too few alternations (streaky, repetitive sequences) look strongly non-random, whereas too many alternations (up to perfect HTHT alternation) are penalised only by a fitted fraction of that slope, because over-alternation is what people expect of chance; this addresses the critique that the incumbent over-penalises perfectly alternating sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/asymmetric_alternation_representativeness.py`

### iter1_candidate4 — rank 4, 3.9 ± 12.0 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -1025.5; PSIS-LOO unreliable (1% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same multiscale local balance plus irregularity score, but their sense of "periodic pattern" only picks up repeating templates longer than two flips (e.g. HHT-HHT, HHTT-HHTT); streaks (period 1) and plain alternation (period 2) are judged only through the balance and alternation-rate cues, not penalised a second time as periodic patterns. The single change is this non-redundant periodicity penalty, addressing the critique that the incumbent over-penalises perfect alternation (and, more weakly, long streaks) relative to what people choose.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter1_candidate4.py`

### adaptive_transition_learner_surprise — rank 5, 7.4 ± 9.6 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1029.0

**Hypothesis:** People judge randomness by reading each sequence flip by flip while an online pattern learner tries to predict whether the next flip will repeat or switch, starting from a prior expectation (which may favour switching) and updating that expectation from the transitions seen so far. A sequence looks random to the extent this learner keeps being surprised — so sequences whose transitions become predictable as you read them (long streaks, but also perfect alternation once it has been seen a few times) look less random — and people differ in how decisively this felt unpredictability drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/adaptive_transition_learner_surprise.py`

### goldilocks_gamblers_surprise_v2 — rank 6, 9.0 ± 9.3 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1030.7

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is a concave ("just-right") function of the recency-weighted surprise rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/goldilocks_gamblers_surprise_v2.py`

### ideal_switch_rate_prototype — rank 7, 11.3 ± 8.9 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -1033.0

**Hypothesis:** People judge randomness by a single gist cue: how often the sequence switches between heads and tails. They hold an internal ideal switch rate (a fair coin's "should look like" rate, which may be above one half), and a sequence looks random to the extent its switch rate is close to that ideal — too few switches (long streaks) and too many switches (perfect alternation) both make it look less random. People differ in how decisively this one cue drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/ideal_switch_rate_prototype.py`

### recency_weighted_gamblers_surprise — rank 8, 13.8 ± 12.2 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1035.4; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness by reading each sequence flip by flip while predicting the next flip with a gambler's-fallacy expectation — the longer the current run, the more they expect it to break — and a sequence looks random to the extent its flips were unsurprising under that expectation. Their memory of the surprise fades, so surprises near the end of the sequence (such as a final repeat that extends a run) weigh more than early ones; people differ in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/recency_weighted_gamblers_surprise.py`

### gamblers_fallacy_leaky_predictor — rank 9, 14.3 ± 14.8 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1035.9; PSIS-LOO unreliable (1% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness by running a gambler's-fallacy predictor through the sequence: before each flip they expect the coin to "correct" the heads/tails imbalance seen so far, with recent flips weighing more than older ones in that leaky running tally, and a sequence looks random to the extent its flips match those corrective expectations (high average predictive probability). Unlike the incumbent's position-blind balance and alternation summaries, this makes the *order* of flips matter — a terminal repeat or a late streak (which violates a strong, just-built expectation of reversal) is penalised far more than the same repeat early on — so the two models disagree most on equal-composition pairs that differ only in where a repeat or run sits, especially at the end; people also differ in how sharply they apply this judgment.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/gamblers_fallacy_leaky_predictor.py`

### streak_tolerance_alarm — rank 10, 48.3 ± 19.7 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1069.9; PSIS-LOO unreliable (4% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness with a streak alarm: there is a tolerance for how long a run of identical flips may be before it "looks too long to be chance", and every run in a sequence that exceeds that tolerance raises the alarm, more so the further it exceeds it, while runs within the tolerance cost nothing. The sequence raising the smaller alarm is chosen as more random, and people differ in how strongly the alarm drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/streak_tolerance_alarm.py`

### motif_stack_person_sensitivity — rank 11, 66.2 ± 22.9 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1087.9

**Hypothesis:** Refinement of `motif_stack`: people judge a sequence random to the extent it is poorly explained by the Griffiths et al. (2018) four-motif stack automaton (mirror, complement and duplication production methods) relative to a fair coin, exactly as in `motif_stack` — but each person applies that shared regularity-detection process with their own decisiveness. The single change is that the sensitivity mapping the randomness-score difference to choice varies across participants (a hierarchical, log-normal population of per-person sensitivities) instead of being one shared value, addressing the critique that people differ in how consistently they agree with the majority judgment far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/motif_stack_person_sensitivity.py`

### local_representativeness — rank 12, 238.3 ± 26.4 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1259.9

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/local_representativeness.py`

### motif_stack — rank 13, 280.9 ± 28.8 nats behind the best (9.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1302.5

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### falk_konold_dp — rank 14, 514.7 ± 33.5 nats behind the best (15.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1536.3

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### finite_experience_occurrence — rank 15, 716.0 ± 68.0 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1737.6

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
