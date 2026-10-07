# Tried before

68 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### mismatch_noise_paired_comparison — pruned (experiment2 end of experiment)

**Outcome:** 303.9 nats behind face_specific_streak_chance_coin (7.7× dse)

**Hypothesis:** People do not evaluate each sequence on its own and then compare two independent impressions; they compare the two sequences flip against flip, so positions where the two show the same outcome cancel out and only the mismatching positions carry evidence — and also noise — into the judgment. Each sequence's randomness evidence is the Bayesian "fair coin (believed to over-alternate) versus regular generator" score, but the noise in comparing them grows with the number of positions at which the two sequences differ, so the same difference in randomness is judged decisively beside a near-identical partner (and on short pairs) and hesitantly beside a very different one.

### most_lopsided_window_alarm — pruned (experiment1 end of experiment)

**Outcome:** 107.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by scanning a sequence for its single most lopsided stretch: among all contiguous windows of every length, they find the one whose heads/tails imbalance would be most surprising for a fair coin, and that one striking stretch alone sets how non-random the sequence looks. The sequence whose worst stretch is less surprising is chosen as more random, so locally balanced sequences (including perfect alternation) look random while any single streak or heavily one-sided patch condemns a sequence regardless of how balanced the rest is; people differ in how decisively this drives their choice.

### switch_prototype_person_lapse — pruned (experiment1 end of experiment)

**Outcome:** 26.1 nats behind length_normalised_chance_vs_motif (4.3× dse)

**Hypothesis:** People all judge randomness the same way — by how close a sequence's rate of switching between heads and tails is to their internal ideal switch rate — and apply that judgment with one shared, sharp decision rule; what differs between people is a lapse rate: on some fraction of trials (person-specific) a participant does not use the judgment at all and picks a side at random. Disagreement with the majority therefore comes from occasional inattentive coin-flip choices, which cap how decisive anyone's choices can be (even for clear-cut pairs such as perfect alternation versus a long streak), rather than from graded differences in sensitivity.

### bayesian_chance_vs_repeating_motif — pruned (experiment2 end of experiment)

**Outcome:** 250.7 nats behind face_specific_streak_chance_coin (7.3× dse)

**Hypothesis:** Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators, with every generator's unknowns averaged out normatively. The single change is one more regular generator in the comparison — a "repeating-pattern" generator that picks a short motif (one to four flips) and repeats it — given a fitted share of the prior on regularity; so exactly periodic sequences (perfect alternation HTHT…, but also HHTHHT…, HHTTHHTT…) are recognised as patterns and condemned, addressing the critique that the incumbent over-credits perfect alternation and under-penalises period-3/4 motifs.

### bayesian_chance_vs_repeating_motif_2 — pruned (experiment2 end of experiment)

**Outcome:** 236.1 nats behind face_specific_streak_chance_coin (6.9× dse)

**Hypothesis:** Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent a fair coin (which they believe over-alternates) explains it better than a "regular" generator, with the regular generators' unknowns averaged out normatively. The single change is one more regular generator in their hypothesis space: a "repeating motif" process that writes a short pattern (one to four flips long, e.g. H, HT, HHT, HHTT) and keeps copying it with an occasional slip (a fitted slip rate), so sequences that are near-repetitions of a short motif — perfect alternation and period-3/4 patterns above all — are explained as regular and look less random, addressing the critique that the incumbent under-penalises perfect alternation and periodic motifs.

### global_local_balance_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 31.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same score — H/T balance plus irregularity relative to an over-alternating prototype and periodic templates — but the grain at which they check balance is fitted rather than fixed. The single change: instead of averaging whole-sequence balance and short-window (2–4 flip) balance with equal fixed weights, people give a fitted share of their balance judgment to the sequence's overall heads/tails balance and the rest to local window balance, addressing the critique that people weight global H/T balance more than current models imply among sequences with similar switch counts.

### tally_excursion_goldilocks — pruned (experiment1 end of experiment)

**Outcome:** 350.7 nats behind length_normalised_chance_vs_motif (9.6× dse)

**Hypothesis:** People judge randomness by keeping a running tally of heads minus tails as they read a sequence, and they expect a fair coin's tally to wander away from balance by a moderate, characteristic amount before drifting back. A sequence looks random to the extent its tally's typical excursion from balance (relative to how far a fair coin's tally should have strayed by each point) matches that expected wandering: a tally pinned at balance (strict alternation) looks contrived, and one that runs far to one side (streaks, lopsided stretches) looks non-random, regardless of the final count.

### designed_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 283.9 nats behind length_normalised_chance_vs_motif (6.6× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously "designed" coin sequences — a streak of one side (HHHH…, TTTT…), perfect alternation (HTHT…, THTH…), and short repeated motifs (HHT…, HTT…, HHTT… repeated) — and a sequence looks random to the extent it is dissimilar from all of them, with similarity falling off exponentially with the number of flips at which the sequence differs from each remembered example (summed over examples, so the nearest ones dominate). The sequence that resembles the designed exemplars less is chosen as more random; how memorable each kind of exemplar is and how steeply similarity falls with mismatches are fitted.

### gist_tail_probability_test — pruned (experiment2 end of experiment)

**Outcome:** 476.0 nats behind face_specific_streak_chance_coin (6.7× dse)

**Hypothesis:** People judge randomness like an intuitive significance test on a sequence's gist rather than by comparing explanations: they summarise each sequence by its head count, its number of switches and its longest run, and ask how often a fair coin (which they believe switches somewhat more than half the time) would produce a gist at least as rare as this one. A sequence looks random to the extent that this tail probability is high — a typical gist is unremarkable, while a rare gist (a lopsided count, a long streak, or too-perfect alternation) "rejects chance" — and people differ in how decisively this felt surprise drives their choice.

### length_normalised_chance_vs_motif — pruned (experiment2 end of experiment)

**Outcome:** 215.3 nats behind face_specific_streak_chance_coin (6.5× dse)

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is in how that evidence is weighed: people judge the evidence per flip rather than in total, so the log evidence difference is divided by a fitted power of the sequence length — the same per-flip regularity is judged about as decisively in a short pair as in a long one, instead of long sequences automatically yielding near-certain choices — addressing the critique that people are more decisive on short pairs (relative to long) than the incumbent predicts and that it over-penalises long perfect alternation.

### length_normalised_chance_vs_motif_2 — rejected (experiment1 round 3 candidate 4 refine incumbent bayesian_chance_vs_repeating_motif_2)

**Outcome:** predicts like existing model 'length_normalised_chance_vs_motif' (p_left RMSE 0.00067 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_normalised_chance_vs_motif, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (Markov switch coin, biased coin, repeating motif with slips), but they weigh that evidence per flip rather than in total. The single change is a length normalisation: the log evidence difference between the two sequences is divided by the sequence length raised to a fitted power, so a long pair does not feel proportionally more decisive than a short one — addressing the critique that people are more decisive on short pairs relative to long ones than the incumbent predicts (and that it over-condemns long perfect alternations).

### mismatch_noise_repeating_motif — pruned (experiment2 end of experiment)

**Outcome:** 237.1 nats behind face_specific_streak_chance_coin (6.9× dse)

**Hypothesis:** Refinement of `mismatch_noise_paired_comparison`: people still compare the two sequences flip against flip, so matching positions cancel and the noise in the comparison grows with the number of positions at which the two sequences differ (making short and near-identical pairs judged more decisively), and each sequence's evidence is still the Bayesian "fair coin believed to over-alternate versus a regular generator" score. The single change is to the regular generators people entertain: besides a switch-biased and a heads-biased coin, they also consider a repeating-motif process that copies a short pattern (one to four flips) with an occasional slip, so near-periodic sequences such as perfect alternation or HHT-HHT are recognised as regular and look less random.

### bayesian_chance_vs_motif_beta_trick_coin — pruned (experiment2 end of experiment)

**Outcome:** 224.2 nats behind face_specific_streak_chance_coin (6.6× dse)

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with each generator's unknowns averaged out. The single change is in what people imagine a "biased coin" to be: instead of a uniform prior over its heads rate, they hold a symmetric prior of fitted concentration — imagining either only grossly lopsided trick coins or mildly biased ones — which sets how much a modest heads/tails imbalance counts as evidence against chance, addressing the critique that people weight heads/tails balance more than the incumbent's biased-coin generator implies among sequences with similar switch counts.

### hot_hand_only_suspicion — pruned (experiment2 end of experiment)

**Outcome:** 253.2 nats behind face_specific_streak_chance_coin (6.9× dse)

**Hypothesis:** People judge a sequence random to the extent a plain fair coin explains it better than the only kinds of "non-random" coin they ever suspect — a hot-hand (streaky) coin that tends to repeat its last outcome by an unknown amount, or a coin biased towards heads or tails by an unknown amount — with those unknowns averaged out and the evidence weighed per flip. Switching is never suspicious in itself, so perfect alternation and period-3/4 patterns count as maximally random: this is where the model disagrees most sharply with the current best model, which condemns them as repeating motifs, while among sequences with similar switch counts it still condemns streaks and heads/tails imbalance.

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

### personal_chance_switch_belief — pruned (experiment2 end of experiment)

**Outcome:** 7.3 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 20

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their own model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip. The single change is that the one distortion of the normative account — the believed switch rate of a fair coin — is held differently by each person, drawn from a population: most people believe chance over-alternates, but some believe chance is streaky, so they credit sequences with fewer switches as more random, addressing the critique that a sizeable minority of participants systematically prefer the sequence with fewer switches.

### person_signed_switch_preference — pruned (experiment2 end of experiment)

**Outcome:** 701.6 nats behind face_specific_streak_chance_coin (8.6× dse)

**Hypothesis:** People judge randomness by a single gist cue — how often a sequence switches between heads and tails — but they disagree about which direction is random: most people take frequent switching as the signature of a fair coin, while some hold the opposite intuition that streaky, clumpy sequences are what real chance looks like. Each person therefore has their own signed preference for switching, drawn from a population that can straddle zero, and picks the sequence whose switch rate their own preference favours.

### posterior_verdict_contrast — pruned (experiment2 end of experiment)

**Outcome:** 218.4 nats behind face_specific_streak_chance_coin (6.6× dse)

**Hypothesis:** People do not compare the two sequences' raw weight of evidence; each sequence is first turned into a felt verdict, "how likely is it that this one came from a fair coin rather than from some regular generator" (a fair coin believed to over-alternate, against a switch-biased coin, a biased coin and a repeating short motif), and the choice is driven by the difference between the two verdicts. Because verdicts saturate, the same difference in evidence is decisive when the two sequences are both ambiguous and barely matters when both clearly look random or both clearly look regular, so a sequence's pull on the choice depends on its partner.

### person_specific_chance_switch_belief — pruned (experiment2 end of experiment)

**Outcome:** 4.4 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 20

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that each person holds their own belief about how often a fair coin switches, drawn from a population distribution, instead of everyone sharing one over-alternating belief — so some people believe chance switches a lot while others believe chance is streaky and choose the sequence with fewer switches, addressing the critique that far more participants prefer fewer-switch sequences than the incumbent's shared belief allows.

### person_specific_chance_switch_belief_2 — pruned (experiment2 end of experiment)

**Outcome:** 24.1 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 20

**Hypothesis:** Refinement of `bayesian_chance_vs_motif_beta_trick_coin`: people still judge a sequence random to the extent their own model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin with a fitted-concentration prior, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is that each person holds their own belief about how often a fair coin switches sides, drawn from a population distribution, instead of everyone sharing one over-alternating belief: most people expect chance to over-alternate, but some believe a fair coin tends to repeat itself, so for them streakier sequences look more random — addressing the critique (surviving FDR) that far more participants prefer the sequence with fewer switches than a shared-belief model can produce.

### person_side_bias_chance_vs_motif — pruned (experiment2 end of experiment)

**Outcome:** 214.0 nats behind face_specific_streak_chance_coin (6.4× dse)

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their over-alternating model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out, the evidence weighed per flip, and person-specific decisiveness. The single change is that each person also has their own habitual leaning towards the left or the right response, drawn from a population, which adds to the randomness evidence before the choice — so pairs with weak evidence are decided largely by a person's side habit and the same evidence yields different choice rates across people, addressing the critique that participants' rates of choosing the left sequence vary more than the incumbent allows.

### longest_patterned_stretch — pruned (experiment2 end of experiment)

**Outcome:** 744.6 nats behind face_specific_streak_chance_coin (13.5× dse)

**Hypothesis:** People judge randomness by the single most striking patterned stretch in a sequence: they scan for the longest contiguous stretch that keeps copying one short motif — a streak (H, H, H…), strict alternation (HT, HT…), or a repeated three- or four-flip motif — and how many flips that one stretch "predicts" sets how non-random the whole sequence looks, whatever the rest of it does. Each kind of motif has its own salience (a run of copies of a streak need not be as damning as an equally long alternation), and the sequence whose most striking patterned stretch is weaker is chosen as more random.

### switch_indifference_side_default — pruned (experiment2 end of experiment)

**Outcome:** 704.5 nats behind face_specific_streak_chance_coin (8.6× dse)

**Hypothesis:** People commit to a "more random" choice only when the two sequences differ clearly in their single gist cue, how often they switch between heads and tails (each person with their own signed preference for switching); when the difference in switch rate falls inside an indifference band, they feel no real preference and fall back on their own habitual response side instead of judging. Side habits therefore show up mainly on pairs with similar switch counts, which makes people's overall rates of choosing Left vary more than a judgment-only model allows.

### tally_reverting_chance_coin — pruned (experiment2 end of experiment)

**Outcome:** 1155.0 nats behind face_specific_streak_chance_coin (8.0× dse)

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails, and they believe a fair coin keeps that tally in check: whenever one side is ahead, the next flip is expected to favour the side that is behind, more strongly the bigger the lead (a gambler's-fallacy pull back to balance acting on the cumulative count, not on the last flip). A sequence looks random to the extent its tally path is probable under this self-correcting coin — so early drifts that are corrected look random, while a lead that keeps growing or is left uncorrected looks non-random even when the final count is balanced — and people differ in how strongly (or even in which direction) they expect the tally to be pulled back.

### person_side_habit_switch_belief — pruned (experiment2 end of experiment)

**Outcome:** 3.6 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 20

**Hypothesis:** Refinement of the incumbent `person_specific_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out, the evidence weighed per flip by a fitted power of length, and person-specific decisiveness. The single change is that each person also has their own habitual leaning towards the left or the right response, drawn from a population, which adds to the randomness evidence before the choice — so when the two sequences look about equally random a person's side habit decides — addressing the critique that participants' proportions of Left choices vary more across people than the incumbent produces.

### person_specific_motif_suspicion — pruned (experiment2 end of experiment)

**Outcome:** 2.7 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 20

**Hypothesis:** Refinement of the incumbent `person_specific_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that people also differ in how readily they suspect a repeating pattern: each person gives the repeating-motif generator their own prior weight among the regular explanations, drawn from a population, so for pattern-suspicious people perfect alternation and period-3/4 motifs look strongly non-random while pattern-blind people judge such sequences mainly by their switches and balance.

### switch_belief_habitual_side_lapse — pruned (experiment2 end of experiment)

**Outcome:** 7.0 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 20

**Hypothesis:** Refinement of `personal_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the evidence weighed per flip and person-specific decisiveness. The single change is that on a fraction of trials people do not make the judgment at all and simply click their own habitual side (each person has their own default side), so each person's choices are pulled towards their preferred button by a fixed share even on clear-cut pairs — unlike an additive side bias, which matters only when the evidence is weak — addressing the critique that participants' individual rates of choosing Left vary more than a model without response habits produces.

### exemplar_random_vs_designed_ratio — pruned (experiment2 end of experiment)

**Outcome:** 902.0 nats behind face_specific_streak_chance_coin (9.2× dse)

**Hypothesis:** People judge randomness by exemplar categorisation: they carry remembered examples of both kinds of sequence — "designed" ones (streaks, perfect alternation, and short three- or four-flip motifs repeated) and "random-looking" ones (balanced heads and tails, no long streak, not perfectly alternating) — and a sequence looks random to the extent its summed similarity to the random exemplars outweighs its summed similarity to the designed exemplars, with similarity falling off exponentially with the number of mismatching flips so that the nearest exemplars dominate. The sequence with the higher random-versus-designed similarity ratio is chosen, with people differing only in how decisively they act on it.

### personal_second_order_chance_typicality — pruned (experiment2 end of experiment)

**Outcome:** 466.8 nats behind face_specific_streak_chance_coin (7.9× dse)

**Hypothesis:** People judge randomness by typicality under their own imagined chance process alone, without weighing any rival "regular" explanation: each person carries a personal second-order picture of how a fair coin behaves — how likely it is to switch right after a repeat, and how likely right after a switch — and a sequence looks random to the extent that this imagined coin would readily produce it (its per-flip probability under that picture). Because the switch expectation depends on whether the previous step was a repeat or a switch, someone who believes chance rarely switches twice in a row condemns long perfect alternation, while someone who believes repeats quickly give way to switches condemns streaks, and people differ in both beliefs.

### heads_rigged_trick_coin_suspicion — pruned (experiment2 end of experiment)

**Outcome:** 3.2 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 20

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators they suspect (a switch-biased Markov coin, a trick coin, and a repeating short motif with slips, with each person's own readiness to suspect a motif), weighing the evidence per flip by a fitted power of length. The single claim is that the trick coin people imagine is not symmetric: they suspect a coin rigged towards heads (the side a cheat would favour) more than one rigged towards tails, so a surplus of heads is taken as evidence of rigging while an equal surplus of tails is not. The model disagrees most sharply with the current best (H/T-symmetric) model on pairs of mirror-image or count-mismatched sequences, e.g. HHTHHH versus TTHTTT, where it predicts the tail-heavy sequence is chosen as more random — addressing the FDR-surviving critique that people choose the sequence with more heads less often than a symmetric model allows.

### asymmetric_trick_coin_motif_suspicion — rejected (experiment2 round 2 candidate 3 refine incumbent person_specific_motif_suspicion)

**Outcome:** predicts like existing model 'heads_rigged_trick_coin_suspicion' (p_left RMSE 0.00018 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heads_rigged_trick_coin_suspicion, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `person_specific_motif_suspicion`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips, with each person's own suspicion of motifs), the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that the trick coin people suspect is not symmetric: they expect a rigged coin to favour heads more (or less) than tails, by a fitted asymmetry in their prior over the trick coin's heads rate, so a head-heavy sequence is more readily explained as rigged and looks less random than the equally lopsided tail-heavy sequence — addressing the critique (surviving FDR) that people choose the sequence with more heads less often than the H/T-symmetric incumbent predicts.

### heads_rigged_coin_suspicion — pruned (experiment2 end of experiment)

**Outcome:** 3.6 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 20

**Hypothesis:** Refinement of `person_side_habit_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the evidence weighed per flip, person-specific decisiveness and a person-specific side habit. The single change is in what people imagine a "biased coin" to be: they do not suspect heads-rigged and tails-rigged coins equally, but hold a lopsided prior (of fitted direction and strength) that trick coins are rigged towards heads, so a head-heavy sequence is more readily explained as coming from a rigged coin and looks less random than the mirror-image tail-heavy sequence — addressing the critique (surviving FDR) that people choose the sequence with more heads less often than an H/T-symmetric model allows.

### position_weighted_switch_impression — pruned (experiment2 end of experiment)

**Outcome:** 2.5 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 20

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate, and a second-order belief about switching twice in a row) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own suspicion of motifs), weighing the evidence per flip by a fitted power of length. The single new claim is about the order of reading: people read the sequence left to right and their felt sense of how often it switches is not position-blind — switches and repeats near the end of the sequence (recency) or near its start (primacy, fitted direction and strength) weigh more in that impression, so two sequences with the same switch count but with their streak at different places look differently random.

### attention_gradient_sequential_evidence — pruned (experiment2 end of experiment)

**Outcome:** 3.8 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 20

**Hypothesis:** People judge randomness by reading a sequence flip by flip and accumulating evidence for "fair coin" against the regular explanations they suspect (a switch-biased coin, a biased coin, a repeating short motif with slips — their own believed switch rate of chance and their own suspicion of motifs included), but their attention is not spread evenly over the sequence: the evidence each flip contributes is weighted by an attention gradient along the reading order whose direction and steepness are fitted (primacy if early flips dominate, recency if late flips do). So the same streak, imbalance or pattern break makes a sequence look more or less random depending on where in the sequence it sits, and the model disagrees with position-blind accounts most on pairs that contain the same flips in a different order (e.g. a streak at the start versus at the end).

### leaky_memory_chance_vs_regular — rejected (experiment2 round 3 candidate 2 lens 4)

**Outcome:** MCMC did not converge (9000 divergent transitions of 12000; max R-hat 9.315 > 1.05; min bulk ESS 4 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge randomness as Bayesian inference — does their own picture of a fair coin (with a person-specific belief about how often it switches) explain the sequence better than regular generators (a switch-biased coin, a biased coin, a short motif repeated with slips) — but they read the sequence flip by flip with a leaky memory. The evidence each flip gives for chance over regularity fades as later flips arrive (or, if the fitted direction is reversed, early flips dominate as a first impression), so where in a sequence a streak, a switch or a pattern break occurs changes how random it looks, not only how many there are.

### occam_neglect_trick_coin_fit — pruned (experiment2 end of experiment)

**Outcome:** 2.1 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 20

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate, and a second-order belief about switching twice in a row) explains it better than the regular generators (a switch-biased coin, a biased coin, and a short motif repeated with slips, with each person's own suspicion of motifs). The single distortion is a neglect of Occam's razor: instead of averaging over every rate a trick coin might have, people partly credit the trick coins with whatever switch rate or heads rate best fits the sequence at hand (a fitted share between the normative average and the best fit), so any sequence whose rates are even mildly lopsided is readily "explained" by a tailor-made trick coin, and moderately imbalanced or moderately streaky sequences look less random than the normative account implies, relative to perfectly typical ones.

### symmetry_generator_suspicion — pruned (experiment2 end of experiment)

**Outcome:** 5.3 nats behind face_specific_streak_chance_coin; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 20

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin explains it better than the regular ways a sequence could have been made — and among those regular ways they entertain a kind no current model considers: a sequence built by symmetry, whose second half is the first half mirrored (a palindrome, HHTTTTHH), mirrored with the faces swapped (HHTHTHTT), or repeated with the faces swapped (HTTHTHHT), copied with occasional slips. So sequences that are globally symmetric look designed and less random, even when their switch count, heads/tails balance and short periodic motifs are unremarkable (everything else as in the current best account: a person-specific believed switch rate and second-order switch belief, a switch-biased coin, a biased coin, a repeating short motif with each person's own motif suspicion, evidence weighed per flip, person-specific side habit).

### first_read_satisficing_anchor — rejected (experiment2 round 4 candidate 1 lens 6)

**Outcome:** predicts like existing model 'second_order_motif_side_habit' (p_left RMSE 0.00122 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of second_order_motif_side_habit, not a new hypothesis.

**Hypothesis:** People do not weigh the two sequences symmetrically: they read the left sequence first and treat it as the default answer, keeping it unless the second sequence beats it, so the choice depends not only on which sequence looks more random but on how random the pair looks overall. When both sequences look convincingly random (each better explained by their own picture of a fair coin than by a regular generator — a switch-biased coin, a biased coin, or a short repeating motif), people settle for the first one they read; when both look regular, they reject the first and move to the second — so the same sequence is chosen more or less often depending on how random its partner looks.

### most_damning_stretch_evidence — rejected (experiment2 round 4 candidate 2 lens 7)

**Outcome:** too slow to fit: the fit of 'most_damning_stretch_evidence' (target_accept 0.8) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People judge randomness by the single most suspicious stretch of a sequence, not by the sequence as a whole: they scan every contiguous stretch of three or more flips, ask how much better a regular generator (a switch-biased coin, a biased coin, or a short motif copied with slips) explains that stretch than their own picture of a fair coin (with a person-specific belief about how often chance switches), and the strongest such case anywhere in the sequence alone sets how non-random it looks. So one streak or one patterned run condemns a sequence even when the rest of it looks perfectly chancy, and the sequence whose worst stretch is less damning is chosen as more random.

### second_order_heads_rigged_side_habit — rejected (experiment2 round 4 candidate 4 refine incumbent second_order_motif_side_habit)

**Outcome:** predicts like existing model 'heads_rigged_second_order_side_habit' (p_left RMSE 0.00024 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heads_rigged_second_order_side_habit, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `second_order_motif_side_habit`: people still judge a sequence random to the extent their own second-order model of a fair coin (a person-specific believed switch rate, and a shared shift in how likely a switch is right after a switch) explains it better than the regular generators (a switch-biased Markov coin, a trick coin, and a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length, with each person's own side habit deciding near-ties. The single change, taken from `heads_rigged_trick_coin_suspicion`, is that the trick coin people suspect is not symmetric: they believe rigged coins are more often rigged towards heads (fitted direction and strength), so a surplus of heads is readily explained away as rigging while an equal surplus of tails is not, and tail-heavy sequences look more random than their head-heavy mirror images — addressing the FDR-surviving critique that people choose the head-heavier sequence less often than the H/T-symmetric incumbent predicts. Unlike a tail-favouring belief about the fair coin itself (linear in the head count), this asymmetry acts only through the rigging explanation, so it matters most for clearly lopsided sequences and fades for near-balanced ones.

### pair_randomness_contrast_gain — rejected (experiment2 round 4 candidate 1 lens 6 repair 1)

**Outcome:** MCMC did not converge (6000 divergent transitions of 12000; max R-hat 3.407 > 1.05; min bulk ESS 4 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People do not judge the two sequences one at a time. The pair as a whole sets a shared reference level of how random-looking it is, and that level sets how sharply the difference between the two is felt. Between two regular-looking sequences (streaks, repeated motifs, lopsided counts) the more random one stands out clearly. Between two sequences that both look random the same difference in randomness is felt only weakly, and choices are closer to a toss-up. So the same sequence is chosen more or less decisively depending on how random its partner looks. Each sequence's randomness is judged as in the current best account. It is the evidence that the person's own picture of a fair coin explains the sequence better than regular generators. That picture has a person-specific believed switch rate and a second-order switch belief. The regular generators are a switch-biased coin, a biased coin, and a repeating short motif with slips, which each person suspects to their own degree. The evidence is weighed per flip, and each person has a side habit.

### worst_stretch_regularity_evidence — pruned (experiment2 end of experiment)

**Outcome:** 650.8 nats behind face_specific_streak_chance_coin (13.3× dse)

**Hypothesis:** People judge randomness by the single most suspicious stretch of a sequence, not by the sequence as a whole: they scan every contiguous stretch of three or more flips, ask how much better a regular generator (a switch-biased coin, a biased coin, or a short motif copied with slips) explains that stretch than their picture of a fair coin (which they believe switches at a fitted rate), and the strongest such case anywhere in the sequence alone sets how non-random it looks. So one streak or one patterned run condemns a sequence even when the rest of it looks perfectly chancy, and the sequence whose worst stretch is less damning is chosen as more random.

### person_lapse_gamblers_second_order — rejected (experiment3 round 0 candidate 4 refine incumbent gamblers_run_second_order_side_habit)

**Outcome:** predicts like existing model 'gamblers_second_order_person_lapse' (p_left RMSE 0.00018 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of gamblers_second_order_person_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `gamblers_run_second_order_side_habit`: people still judge a sequence random to the extent their own second-order, gambler's-fallacy model of a fair coin explains it better than the regular generators they suspect (a switch-biased coin, a biased coin, a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length, with person-specific sensitivity and side habit. The single change is that people also differ in how often they disengage: on a person-specific share of trials (drawn from a population) a participant does not judge the pair at all and picks a side at random, which caps how consistently that person agrees with the majority even on clear-cut pairs — addressing the critique that people differ in their rate of agreeing with the majority more than graded sensitivity alone produces (a subgroup of near-random responders).

### signed_sensitivity_person_lapse — rejected (experiment3 round 1 candidate 4 refine incumbent gamblers_second_order_person_lapse)

**Outcome:** MCMC did not converge (307 divergent transitions of 12000; min bulk ESS 65 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of the incumbent `gamblers_second_order_person_lapse` (people judge a sequence random to the extent their own second-order, gambler's-fallacy model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a biased coin, a repeating short motif with slips — weighed per flip by a fitted power of length, with person-specific side habits and person-specific random lapses). The single change is that people do not all read regularity as a sign of non-randomness: each person's sensitivity to the "regular generator versus chance" evidence is drawn from a population that can straddle zero instead of being forced positive, so a minority of participants systematically treat patterned, streaky or periodic sequences as the more random-looking ones (or invert the judgment) and agree with the majority below chance even on clear-cut pairs — addressing the FDR-surviving critique that participants differ in their rate of agreeing with the majority more than the incumbent's lapses and positive sensitivities produce.

### signed_commitment_heads_rigged_motif — rejected (experiment3 round 1 candidate 5 refine chosen)

**Outcome:** MCMC did not converge (max R-hat 1.639 > 1.05; min bulk ESS 6 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of `attentive_lapse_heads_rigged_motif` (people judge a sequence random to the extent their own second-order model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a heads-favoured trick coin and a repeating motif with slips — the evidence weighed per flip, with person-specific sensitivity and side habit, and a person-specific share of trials answered at random). The single change is that a person's departure from their own judgment is not capped at random guessing: each person's commitment to their randomness impression is drawn from a population that is mostly near full commitment but can fall below zero, so a few participants systematically pick the sequence that looks *less* random to everyone else (having reversed the question), agreeing with the majority well below chance — in the data four participants agree on only 22–41% of trials, which no lapse rate can produce — addressing the FDR-surviving critique that participants' agreement with the majority varies across people more than the lapse model produces.

### heavy_tailed_sensitivity_person_lapse — rejected (experiment3 round 1 candidate 4 refine incumbent gamblers_second_order_person_lapse repair 1)

**Outcome:** predicts like existing model 'gamblers_second_order_person_lapse' (p_left RMSE 0.00060 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of gamblers_second_order_person_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `gamblers_second_order_person_lapse` (people judge a sequence random to the extent their own second-order, gambler's-fallacy model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a biased coin, a repeating short motif with slips — weighed per flip by a fitted power of length, with person-specific side habits and person-specific random lapses). The single change is in how people differ in decisiveness: each person's sensitivity to that evidence comes from a heavy-tailed population rather than a bell-shaped one, so while most people act on the evidence about equally, a few are almost perfectly consistent and a few barely act on it at all — making people's rates of agreeing with the majority spread more than graded sensitivity plus lapses produces (the FDR-surviving critique).
