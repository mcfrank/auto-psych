# Tried before

29 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### negative_recency_expectation_fit — rejected (experiment1 round 0 candidate 0 lens 0)

**Outcome:** MCMC did not converge (401 divergent transitions of 12000), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge randomness by reading each sequence flip by flip and checking how well every flip matches a gambler's-fallacy expectation: after a streak, they expect the coin to switch, increasingly so the longer the streak has run. A sequence looks random to the extent its flips conform to this negative-recency expectation, with the most recently read flips (near the end of the sequence) weighing more heavily, and people differ in how strongly this conformity drives their choice.

### gamblers_fallacy_leaky_predictor — pruned (experiment1 end of experiment)

**Outcome:** 43.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by running a gambler's-fallacy predictor through the sequence: before each flip they expect the coin to "correct" the heads/tails imbalance seen so far, with recent flips weighing more than older ones in that leaky running tally, and a sequence looks random to the extent its flips match those corrective expectations (high average predictive probability). Unlike the incumbent's position-blind balance and alternation summaries, this makes the *order* of flips matter — a terminal repeat or a late streak (which violates a strong, just-built expectation of reversal) is penalised far more than the same repeat early on — so the two models disagree most on equal-composition pairs that differ only in where a repeat or run sits, especially at the end; people also differ in how sharply they apply this judgment.

### recency_weighted_gamblers_surprise — pruned (experiment1 end of experiment)

**Outcome:** 43.2 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by reading each sequence flip by flip while predicting the next flip with a gambler's-fallacy expectation — the longer the current run, the more they expect it to break — and a sequence looks random to the extent its flips were unsurprising under that expectation. Their memory of the surprise fades, so surprises near the end of the sequence (such as a final repeat that extends a run) weigh more than early ones; people differ in how strongly this felt surprise drives their choice.

### iter0_candidate3 — pruned (experiment1 end of experiment)

**Outcome:** 30.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 24

**Hypothesis:** Refinement of `local_representativeness`: people judge randomness by the same Kahneman & Tversky local-representativeness score (multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates), but they differ in how decisively they apply it. The single change is that the decision sensitivity (beta) is person-specific, drawn from a population distribution, rather than shared by everyone — addressing the critique that people differ in agreement with the majority far more than one pooled sensitivity allows.

### iter0_candidate4 — pruned (experiment1 end of experiment)

**Outcome:** 31.2 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 24

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random to the extent it is locally balanced at several window scales and irregular, i.e. somewhat over-alternating but not periodic). The one change: people share this representativeness criterion but differ in how consistently they apply it, so each participant has their own decision sensitivity (drawn from a population distribution) rather than one shared sensitivity — addressing the critique that participants differ in agreement with the majority far more than a single pooled sensitivity allows.

### motif_stack_person_sensitivity — pruned (experiment1 end of experiment)

**Outcome:** 95.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 23 of 24

**Hypothesis:** Refinement of `motif_stack`: people judge a sequence random to the extent it is poorly explained by the Griffiths et al. (2018) four-motif stack automaton (mirror, complement and duplication production methods) relative to a fair coin, exactly as in `motif_stack` — but each person applies that shared regularity-detection process with their own decisiveness. The single change is that the sensitivity mapping the randomness-score difference to choice varies across participants (a hierarchical, log-normal population of per-person sensitivities) instead of being one shared value, addressing the critique that people differ in how consistently they agree with the majority judgment far more than a single pooled sensitivity allows.

### streak_tolerance_alarm — pruned (experiment1 end of experiment)

**Outcome:** 77.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness with a streak alarm: there is a tolerance for how long a run of identical flips may be before it "looks too long to be chance", and every run in a sequence that exceeds that tolerance raises the alarm, more so the further it exceeds it, while runs within the tolerance cost nothing. The sequence raising the smaller alarm is chosen as more random, and people differ in how strongly the alarm drives their choice.

### adaptive_transition_learner_surprise — pruned (experiment1 end of experiment)

**Outcome:** 33.9 nats behind length_normalised_chance_vs_motif (4.6× dse)

**Hypothesis:** People judge randomness by reading each sequence flip by flip while an online pattern learner tries to predict whether the next flip will repeat or switch, starting from a prior expectation (which may favour switching) and updating that expectation from the transitions seen so far. A sequence looks random to the extent this learner keeps being surprised — so sequences whose transitions become predictable as you read them (long streaks, but also perfect alternation once it has been seen a few times) look less random — and people differ in how decisively this felt unpredictability drives their choice.

### bayesian_overalternating_chance_model — pruned (experiment1 end of experiment)

**Outcome:** 29.4 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 24

**Hypothesis:** People judge randomness as Bayesian inference: a sequence looks random to the extent a fair coin explains it better than a "regular" generator (a coin with an unknown tendency to switch or repeat, or a coin with an unknown bias towards heads or tails), with the regular generators' unknown rates averaged out normatively. The one distortion is in their model of chance itself: people believe a fair coin switches sides more often than half the time, so their "random" likelihood rewards alternation and penalises repeats by a fitted amount — which also sets how much a perfectly alternating sequence is still credited as random rather than condemned as regular.

### ideal_switch_rate_prototype — pruned (experiment1 end of experiment)

**Outcome:** 37.8 nats behind length_normalised_chance_vs_motif (4.8× dse)

**Hypothesis:** People judge randomness by a single gist cue: how often the sequence switches between heads and tails. They hold an internal ideal switch rate (a fair coin's "should look like" rate, which may be above one half), and a sequence looks random to the extent its switch rate is close to that ideal — too few switches (long streaks) and too many switches (perfect alternation) both make it look less random. People differ in how decisively this one cue drives their choice.

### asymmetric_alternation_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 31.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific sensitivity): people judge randomness by the same score — multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates — but the deviation from their prototype alternation rate is felt asymmetrically. The single change: too few alternations (streaky, repetitive sequences) look strongly non-random, whereas too many alternations (up to perfect HTHT alternation) are penalised only by a fitted fraction of that slope, because over-alternation is what people expect of chance; this addresses the critique that the incumbent over-penalises perfectly alternating sequences.

### iter1_candidate4 — pruned (experiment1 end of experiment)

**Outcome:** 33.3 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same multiscale local balance plus irregularity score, but their sense of "periodic pattern" only picks up repeating templates longer than two flips (e.g. HHT-HHT, HHTT-HHTT); streaks (period 1) and plain alternation (period 2) are judged only through the balance and alternation-rate cues, not penalised a second time as periodic patterns. The single change is this non-redundant periodicity penalty, addressing the critique that the incumbent over-penalises perfect alternation (and, more weakly, long streaks) relative to what people choose.

### goldilocks_gamblers_surprise — rejected (experiment1 round 1 candidate 5 refine chosen)

**Outcome:** too slow to fit: the fit of 'goldilocks_gamblers_surprise' (target_accept 0.8) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is the closeness of the recency-weighted surprise to a fitted "just-right" level rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.

### goldilocks_gamblers_surprise_v2 — pruned (experiment1 end of experiment)

**Outcome:** 38.5 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 24

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is a concave ("just-right") function of the recency-weighted surprise rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.

### most_lopsided_window_alarm — pruned (experiment1 end of experiment)

**Outcome:** 107.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by scanning a sequence for its single most lopsided stretch: among all contiguous windows of every length, they find the one whose heads/tails imbalance would be most surprising for a fair coin, and that one striking stretch alone sets how non-random the sequence looks. The sequence whose worst stretch is less surprising is chosen as more random, so locally balanced sequences (including perfect alternation) look random while any single streak or heavily one-sided patch condemns a sequence regardless of how balanced the rest is; people differ in how decisively this drives their choice.

### switch_prototype_person_lapse — pruned (experiment1 end of experiment)

**Outcome:** 26.1 nats behind length_normalised_chance_vs_motif (4.3× dse)

**Hypothesis:** People all judge randomness the same way — by how close a sequence's rate of switching between heads and tails is to their internal ideal switch rate — and apply that judgment with one shared, sharp decision rule; what differs between people is a lapse rate: on some fraction of trials (person-specific) a participant does not use the judgment at all and picks a side at random. Disagreement with the majority therefore comes from occasional inattentive coin-flip choices, which cap how decisive anyone's choices can be (even for clear-cut pairs such as perfect alternation versus a long streak), rather than from graded differences in sensitivity.

### global_local_balance_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 31.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same score — H/T balance plus irregularity relative to an over-alternating prototype and periodic templates — but the grain at which they check balance is fitted rather than fixed. The single change: instead of averaging whole-sequence balance and short-window (2–4 flip) balance with equal fixed weights, people give a fitted share of their balance judgment to the sequence's overall heads/tails balance and the rest to local window balance, addressing the critique that people weight global H/T balance more than current models imply among sequences with similar switch counts.

### tally_excursion_goldilocks — pruned (experiment1 end of experiment)

**Outcome:** 350.7 nats behind length_normalised_chance_vs_motif (9.6× dse)

**Hypothesis:** People judge randomness by keeping a running tally of heads minus tails as they read a sequence, and they expect a fair coin's tally to wander away from balance by a moderate, characteristic amount before drifting back. A sequence looks random to the extent its tally's typical excursion from balance (relative to how far a fair coin's tally should have strayed by each point) matches that expected wandering: a tally pinned at balance (strict alternation) looks contrived, and one that runs far to one side (streaks, lopsided stretches) looks non-random, regardless of the final count.

### designed_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 283.9 nats behind length_normalised_chance_vs_motif (6.6× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously "designed" coin sequences — a streak of one side (HHHH…, TTTT…), perfect alternation (HTHT…, THTH…), and short repeated motifs (HHT…, HTT…, HHTT… repeated) — and a sequence looks random to the extent it is dissimilar from all of them, with similarity falling off exponentially with the number of flips at which the sequence differs from each remembered example (summed over examples, so the nearest ones dominate). The sequence that resembles the designed exemplars less is chosen as more random; how memorable each kind of exemplar is and how steeply similarity falls with mismatches are fitted.

### length_normalised_chance_vs_motif_2 — rejected (experiment1 round 3 candidate 4 refine incumbent bayesian_chance_vs_repeating_motif_2)

**Outcome:** predicts like existing model 'length_normalised_chance_vs_motif' (p_left RMSE 0.00067 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_normalised_chance_vs_motif, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (Markov switch coin, biased coin, repeating motif with slips), but they weigh that evidence per flip rather than in total. The single change is a length normalisation: the log evidence difference between the two sequences is divided by the sequence length raised to a fitted power, so a long pair does not feel proportionally more decisive than a short one — addressing the critique that people are more decisive on short pairs relative to long ones than the incumbent predicts (and that it over-condemns long perfect alternations).

### sequential_pattern_alarm_survival — pruned (experiment1 end of experiment)

**Outcome:** 60.9 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness with a sequential pattern alarm: while reading a sequence flip by flip, at each flip they may notice "this is a pattern", with a detection probability that rises with how long the current predictable stretch has lasted — a run of identical flips, or a chain of strict alternation — and alternation chains must persist longer before they trigger the alarm than streaks do, because people expect a fair coin to switch often. A sequence looks random to the extent it can be read to the end without the alarm firing, so a single long streak condemns a sequence even when its overall switch count is ordinary, while long perfect alternation is condemned only weakly; people differ in how decisively this drives their choice.

### gamblers_chance_vs_motif — pruned (experiment1 end of experiment)

**Outcome:** 0.9 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips, unknowns averaged out), with the evidence weighed per flip by a fitted power of sequence length and person-specific sensitivity. The single change is to their model of chance itself: instead of a fair coin that switches with one fixed (over-alternating) probability, people hold a gambler's-fallacy coin whose chance of switching grows with the length of the current run, so under "chance" a long streak is improbable beyond what its switch count implies, while strict alternation (every run of length one) earns only the baseline switch rate — addressing the critique that the incumbent under-penalises long streaks among sequences with equal switch counts and over-credits perfect alternation.

### run_length_gamblers_chance_vs_motif — pruned (experiment1 end of experiment)

**Outcome:** 0.0 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is in their model of chance itself: instead of a fixed belief that a fair coin switches more often than half the time, people hold a gambler's-fallacy belief that the coin becomes more likely to switch the longer the current run has lasted (a fitted increase per extra flip in the run), so a long streak is less probable under "chance" than the same number of switches spread as short runs, and runs of length one (alternation) are what chance is expected to keep producing — addressing the critique that the incumbent under-penalises long streaks among sequences with equal switch counts and over-penalises perfect alternation.

### simplicity_weighted_motif_chance — pruned (experiment1 end of experiment)

**Outcome:** 7.1 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif copied with occasional slips), with the generators' unknowns averaged out. The single change is in which repeating motifs people consider likely: instead of treating every motif period (1 to 4 flips) as equally plausible, they hold a simplicity prior whose fitted decay makes shorter periods — above all a single repeated flip, i.e. a streak — more expected regular patterns than alternation or longer motifs, so long streaks are condemned more and perfect alternation less, addressing the critique that the best model over-penalises perfect alternation and under-penalises long runs among sequences with similar switch counts.

### block_frequency_equidistribution — pruned (experiment1 end of experiment)

**Outcome:** 294.3 nats behind length_normalised_chance_vs_motif (7.2× dse)

**Hypothesis:** People judge randomness by checking whether every short pattern turns up about equally often: they tally the single flips (H, T), the overlapping pairs (HH, HT, TH, TT) and the overlapping triples within a sequence, and a sequence looks random to the extent these tallies are evenly spread (high block entropy relative to the most even spread the sequence's length allows), with shorter blocks weighing more than longer ones by a fitted decay. Streaks fail because one pair/triple dominates, perfect alternation fails only on pairs and triples (it still has perfectly balanced H/T), and lopsided head counts fail at every block size.

### falk_konold_dp — pruned (experiment1 end of experiment)

**Outcome:** 541.2 nats behind length_normalised_chance_vs_motif (15.2× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

### motif_stack — pruned (experiment1 end of experiment)

**Outcome:** 307.4 nats behind length_normalised_chance_vs_motif (10.4× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Outcome:** 742.5 nats behind length_normalised_chance_vs_motif (11.1× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

### local_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 264.8 nats behind length_normalised_chance_vs_motif (10.6× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.
