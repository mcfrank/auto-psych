# Refinement menu

The models you may refine, other than the incumbent `bayesian_chance_vs_repeating_motif_2`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bayesian_chance_vs_repeating_motif — rank 1, 13.3 ± 3.7 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1016.3

**Hypothesis:** Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators, with every generator's unknowns averaged out normatively. The single change is one more regular generator in the comparison — a "repeating-pattern" generator that picks a short motif (one to four flips) and repeats it — given a fitted share of the prior on regularity; so exactly periodic sequences (perfect alternation HTHT…, but also HHTHHT…, HHTTHHTT…) are recognised as patterns and condemned, addressing the critique that the incumbent over-credits perfect alternation and under-penalises period-3/4 motifs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/bayesian_chance_vs_repeating_motif.py`

### switch_prototype_person_lapse — rank 2, 18.3 ± 8.2 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1021.3

**Hypothesis:** People all judge randomness the same way — by how close a sequence's rate of switching between heads and tails is to their internal ideal switch rate — and apply that judgment with one shared, sharp decision rule; what differs between people is a lapse rate: on some fraction of trials (person-specific) a participant does not use the judgment at all and picks a side at random. Disagreement with the majority therefore comes from occasional inattentive coin-flip choices, which cap how decisive anyone's choices can be (even for clear-cut pairs such as perfect alternation versus a long streak), rather than from graded differences in sensitivity.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/switch_prototype_person_lapse.py`

### mismatch_noise_paired_comparison — rank 3, 18.6 ± 6.9 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1021.6

**Hypothesis:** People do not evaluate each sequence on its own and then compare two independent impressions; they compare the two sequences flip against flip, so positions where the two show the same outcome cancel out and only the mismatching positions carry evidence — and also noise — into the judgment. Each sequence's randomness evidence is the Bayesian "fair coin (believed to over-alternate) versus regular generator" score, but the noise in comparing them grows with the number of positions at which the two sequences differ, so the same difference in randomness is judged decisively beside a near-identical partner (and on short pairs) and hesitantly beside a very different one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/mismatch_noise_paired_comparison.py`

### bayesian_overalternating_chance_model — rank 4, 18.6 ± 6.7 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1021.6

**Hypothesis:** People judge randomness as Bayesian inference: a sequence looks random to the extent a fair coin explains it better than a "regular" generator (a coin with an unknown tendency to switch or repeat, or a coin with an unknown bias towards heads or tails), with the regular generators' unknown rates averaged out normatively. The one distortion is in their model of chance itself: people believe a fair coin switches sides more often than half the time, so their "random" likelihood rewards alternation and penalises repeats by a fitted amount — which also sets how much a perfectly alternating sequence is still credited as random rather than condemned as regular.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/bayesian_overalternating_chance_model.py`

### iter0_candidate3 — rank 5, 20.0 ± 11.6 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1023.0

**Hypothesis:** Refinement of `local_representativeness`: people judge randomness by the same Kahneman & Tversky local-representativeness score (multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates), but they differ in how decisively they apply it. The single change is that the decision sensitivity (beta) is person-specific, drawn from a population distribution, rather than shared by everyone — addressing the critique that people differ in agreement with the majority far more than one pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter0_candidate3.py`

### iter0_candidate4 — rank 6, 20.4 ± 11.6 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1023.4

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random to the extent it is locally balanced at several window scales and irregular, i.e. somewhat over-alternating but not periodic). The one change: people share this representativeness criterion but differ in how consistently they apply it, so each participant has their own decision sensitivity (drawn from a population distribution) rather than one shared sensitivity — addressing the critique that participants differ in agreement with the majority far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter0_candidate4.py`

### asymmetric_alternation_representativeness — rank 7, 21.0 ± 12.3 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1024.0; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific sensitivity): people judge randomness by the same score — multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates — but the deviation from their prototype alternation rate is felt asymmetrically. The single change: too few alternations (streaky, repetitive sequences) look strongly non-random, whereas too many alternations (up to perfect HTHT alternation) are penalised only by a fitted fraction of that slope, because over-alternation is what people expect of chance; this addresses the critique that the incumbent over-penalises perfectly alternating sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/asymmetric_alternation_representativeness.py`

### global_local_balance_representativeness — rank 8, 21.0 ± 11.6 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1024.0; PSIS-LOO unreliable (1% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** Refinement of `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same score — H/T balance plus irregularity relative to an over-alternating prototype and periodic templates — but the grain at which they check balance is fitted rather than fixed. The single change: instead of averaging whole-sequence balance and short-window (2–4 flip) balance with equal fixed weights, people give a fitted share of their balance judgment to the sequence's overall heads/tails balance and the rest to local window balance, addressing the critique that people weight global H/T balance more than current models imply among sequences with similar switch counts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/global_local_balance_representativeness.py`

### iter1_candidate4 — rank 9, 22.5 ± 11.6 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1025.5; PSIS-LOO unreliable (1% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same multiscale local balance plus irregularity score, but their sense of "periodic pattern" only picks up repeating templates longer than two flips (e.g. HHT-HHT, HHTT-HHTT); streaks (period 1) and plain alternation (period 2) are judged only through the balance and alternation-rate cues, not penalised a second time as periodic patterns. The single change is this non-redundant periodicity penalty, addressing the critique that the incumbent over-penalises perfect alternation (and, more weakly, long streaks) relative to what people choose.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/iter1_candidate4.py`

### adaptive_transition_learner_surprise — rank 10, 26.0 ± 10.0 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1029.0

**Hypothesis:** People judge randomness by reading each sequence flip by flip while an online pattern learner tries to predict whether the next flip will repeat or switch, starting from a prior expectation (which may favour switching) and updating that expectation from the transitions seen so far. A sequence looks random to the extent this learner keeps being surprised — so sequences whose transitions become predictable as you read them (long streaks, but also perfect alternation once it has been seen a few times) look less random — and people differ in how decisively this felt unpredictability drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/adaptive_transition_learner_surprise.py`

### goldilocks_gamblers_surprise_v2 — rank 11, 27.7 ± 10.0 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1030.7

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is a concave ("just-right") function of the recency-weighted surprise rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/goldilocks_gamblers_surprise_v2.py`

### ideal_switch_rate_prototype — rank 12, 30.0 ± 10.0 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1033.0

**Hypothesis:** People judge randomness by a single gist cue: how often the sequence switches between heads and tails. They hold an internal ideal switch rate (a fair coin's "should look like" rate, which may be above one half), and a sequence looks random to the extent its switch rate is close to that ideal — too few switches (long streaks) and too many switches (perfect alternation) both make it look less random. People differ in how decisively this one cue drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/ideal_switch_rate_prototype.py`

### recency_weighted_gamblers_surprise — rank 13, 32.4 ± 14.1 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1035.4; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness by reading each sequence flip by flip while predicting the next flip with a gambler's-fallacy expectation — the longer the current run, the more they expect it to break — and a sequence looks random to the extent its flips were unsurprising under that expectation. Their memory of the surprise fades, so surprises near the end of the sequence (such as a final repeat that extends a run) weigh more than early ones; people differ in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/recency_weighted_gamblers_surprise.py`

### gamblers_fallacy_leaky_predictor — rank 14, 32.9 ± 15.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1035.9; PSIS-LOO unreliable (1% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness by running a gambler's-fallacy predictor through the sequence: before each flip they expect the coin to "correct" the heads/tails imbalance seen so far, with recent flips weighing more than older ones in that leaky running tally, and a sequence looks random to the extent its flips match those corrective expectations (high average predictive probability). Unlike the incumbent's position-blind balance and alternation summaries, this makes the *order* of flips matter — a terminal repeat or a late streak (which violates a strong, just-built expectation of reversal) is penalised far more than the same repeat early on — so the two models disagree most on equal-composition pairs that differ only in where a repeat or run sits, especially at the end; people also differ in how sharply they apply this judgment.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/gamblers_fallacy_leaky_predictor.py`

### streak_tolerance_alarm — rank 15, 67.0 ± 22.2 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1069.9; PSIS-LOO unreliable (4% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness with a streak alarm: there is a tolerance for how long a run of identical flips may be before it "looks too long to be chance", and every run in a sequence that exceeds that tolerance raises the alarm, more so the further it exceeds it, while runs within the tolerance cost nothing. The sequence raising the smaller alarm is chosen as more random, and people differ in how strongly the alarm drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/streak_tolerance_alarm.py`

### motif_stack_person_sensitivity — rank 16, 84.9 ± 22.6 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1087.9

**Hypothesis:** Refinement of `motif_stack`: people judge a sequence random to the extent it is poorly explained by the Griffiths et al. (2018) four-motif stack automaton (mirror, complement and duplication production methods) relative to a fair coin, exactly as in `motif_stack` — but each person applies that shared regularity-detection process with their own decisiveness. The single change is that the sensitivity mapping the randomness-score difference to choice varies across participants (a hierarchical, log-normal population of per-person sensitivities) instead of being one shared value, addressing the critique that people differ in how consistently they agree with the majority judgment far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/motif_stack_person_sensitivity.py`

### most_lopsided_window_alarm — rank 17, 97.0 ± 24.4 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1100.0; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People judge randomness by scanning a sequence for its single most lopsided stretch: among all contiguous windows of every length, they find the one whose heads/tails imbalance would be most surprising for a fair coin, and that one striking stretch alone sets how non-random the sequence looks. The sequence whose worst stretch is less surprising is chosen as more random, so locally balanced sequences (including perfect alternation) look random while any single streak or heavily one-sided patch condemns a sequence regardless of how balanced the rest is; people differ in how decisively this drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/most_lopsided_window_alarm.py`

### local_representativeness — rank 18, 256.9 ± 25.1 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1259.9

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/local_representativeness.py`

### motif_stack — rank 19, 299.5 ± 28.5 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1302.5

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### falk_konold_dp — rank 20, 533.3 ± 35.3 nats behind the best (15.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1536.3

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### finite_experience_occurrence — rank 21, 734.6 ± 66.7 nats behind the best (11.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1737.6

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

### tally_excursion_goldilocks — no comparison row

**Hypothesis:** People judge randomness by keeping a running tally of heads minus tails as they read a sequence, and they expect a fair coin's tally to wander away from balance by a moderate, characteristic amount before drifting back. A sequence looks random to the extent its tally's typical excursion from balance (relative to how far a fair coin's tally should have strayed by each point) matches that expected wandering: a tally pinned at balance (strict alternation) looks contrived, and one that runs far to one side (streaks, lopsided stretches) looks non-random, regardless of the final count.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/tally_excursion_goldilocks.py`

### designed_exemplar_similarity — no comparison row

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously "designed" coin sequences — a streak of one side (HHHH…, TTTT…), perfect alternation (HTHT…, THTH…), and short repeated motifs (HHT…, HTT…, HHTT… repeated) — and a sequence looks random to the extent it is dissimilar from all of them, with similarity falling off exponentially with the number of flips at which the sequence differs from each remembered example (summed over examples, so the nearest ones dominate). The sequence that resembles the designed exemplars less is chosen as more random; how memorable each kind of exemplar is and how steeply similarity falls with mismatches are fitted.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/designed_exemplar_similarity.py`

### gist_tail_probability_test — no comparison row

**Hypothesis:** People judge randomness like an intuitive significance test on a sequence's gist rather than by comparing explanations: they summarise each sequence by its head count, its number of switches and its longest run, and ask how often a fair coin (which they believe switches somewhat more than half the time) would produce a gist at least as rare as this one. A sequence looks random to the extent that this tail probability is high — a typical gist is unremarkable, while a rare gist (a lopsided count, a long streak, or too-perfect alternation) "rejects chance" — and people differ in how decisively this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/gist_tail_probability_test.py`

### length_normalised_chance_vs_motif — no comparison row

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is in how that evidence is weighed: people judge the evidence per flip rather than in total, so the log evidence difference is divided by a fitted power of the sequence length — the same per-flip regularity is judged about as decisively in a short pair as in a long one, instead of long sequences automatically yielding near-certain choices — addressing the critique that people are more decisive on short pairs (relative to long) than the incumbent predicts and that it over-penalises long perfect alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/length_normalised_chance_vs_motif.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
