# Tried before

41 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_switch_belief — pruned (experiment1 end of experiment)

**Outcome:** 159.2 nats behind individual_streak_aversion_lapse (6.0× dse)

**Hypothesis:** Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

### personal_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 77.5 nats behind individual_streak_aversion_lapse (5.0× dse)

**Hypothesis:** Each person carries their own ideal switching rate for a random coin (how often consecutive flips should differ), and judges a sequence as more random the closer its proportion of alternations lies to that personal ideal; these ideals differ between people, some expecting near-perfect alternation and others streakier sequences. The model disagrees most with the current best model on highly alternating sequences (e.g. HTHTHTHT versus a streaky sequence), where it predicts a population split — some people strongly prefer them, others strongly reject them — and on pairs that differ in balance but not in alternation rate, where it predicts indifference.

### personal_switch_rate_prototype — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_ideal_alternation' (p_left RMSE 0.00008 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_ideal_alternation, not a new hypothesis.

**Hypothesis:** Each person carries their own prototype of how often a genuinely random coin switches between heads and tails, and judges a sequence as random to the extent its switch rate is close to that personal prototype. People differ substantially in this prototype (some expect heavy alternation, others expect near-independent switching), so the same pair can be judged in opposite directions by different participants.

### individual_alternation_prototype — pruned (experiment1 end of experiment)

**Outcome:** 117.7 nats behind individual_streak_aversion_lapse (6.0× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular, i.e. close to a prototype alternation rate and not periodic). The one change: each person holds their own prototype alternation rate — some expect a random sequence to switch much more often than a fair coin does, others barely more or even less — drawn from a population distribution, instead of everyone sharing one prototype. This addresses the critique that people differ far more in how strongly they prefer the more-alternating sequence than a single-population model produces.

### person_prototype_representativeness — rejected (experiment1 round 0 candidate 4 refine incumbent local_representativeness)

**Outcome:** predicts like existing model 'individual_alternation_prototype' (p_left RMSE 0.00063 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_alternation_prototype, not a new hypothesis.

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge a sequence random when it is locally balanced and irregular, but the alternation rate each person's mental prototype of randomness expects differs between people — some expect strong over-alternation, others expect streakier sequences — instead of one shared over-alternating prototype. The single change is that the prototype alternation rate becomes a per-participant parameter (free to lie below or above 0.5) drawn from a population distribution, addressing the critique that real individual differences in alternation preference are far larger than the single-population model produces.

### motif_stack_alternation_individual — pruned (experiment1 end of experiment)

**Outcome:** 112.4 nats behind individual_streak_aversion_lapse (4.7× dse)

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the log-likelihood ratio of a fair coin versus Griffiths et al.'s four-motif stack automaton (the automaton held at the values the motif_stack fit settled on for this data), but people differ in how much they additionally favour or disfavour alternation — each participant carries their own preference for the sequence that switches between H and T more often, drawn from a population distribution. The single change is this participant-level alternation-preference random effect, added because the critique shows the between-participant spread in choosing the more-alternating sequence (SD 0.22) is four times what a single-population model produces.

### online_transition_surprise — pruned (experiment1 end of experiment)

**Outcome:** 624.4 nats behind individual_streak_aversion_lapse (12.3× dse)

**Hypothesis:** People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

### individual_irregularity_weight — pruned (experiment1 end of experiment)

**Outcome:** 148.5 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

### windowed_glimpse_expectation — pruned (experiment1 end of experiment)

**Outcome:** 77.2 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** People read a coin-flip sequence through a limited working-memory window, holding only the last few flips at a time, and compare how mixed each glimpse is (its balance of heads and tails) with how mixed they expect a glimpse of a random coin to be; a sequence looks random to the extent its glimpses match that expectation. The expectation comes from each person's own belief about how often a random coin switches sides, so some people expect near-perfect mixing and others streakier glimpses, and because the window spans more than two flips, lopsided heads/tails counts and long runs are judged beyond what the alternation rate alone shows.

### iter1_candidate1 — pruned (experiment1 end of experiment)

**Outcome:** 82.0 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent it is better explained by a fair coin than by the non-random generators one might suspect — a biased coin (which explains unbalanced heads/tails counts) or a sticky/switchy coin (which explains long runs or rigid alternation). The one distortion is that each person's mental model of a "fair" coin switches between heads and tails at their own personal rate rather than exactly half the time, so people disagree about alternation while all still penalise imbalance and runs that the alternative generators explain.

### personal_ideal_longest_run — pruned (experiment1 end of experiment)

**Outcome:** 133.1 nats behind individual_streak_aversion_lapse (5.0× dse)

**Hypothesis:** People judge randomness by the longest streak a sequence contains: each person carries their own ideal length for the longest run of identical flips in a random sequence (relative to its length), and picks the sequence whose longest streak is closer to that personal ideal. Some people expect random sequences to contain almost no streaks, others expect a noticeable streak, so the same pair can be judged in opposite directions; the rest of the sequence's structure (its overall alternation rate or H/T balance) plays no role beyond what it does to the longest streak.

### ideal_alternation_with_balance — pruned (experiment1 end of experiment)

**Outcome:** 77.1 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people additionally expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how it alternates. The one change is this shared penalty on H/T imbalance, added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which the alternation-only model predicts as indifference.

### balanced_personal_ideal_alternation — rejected (experiment1 round 1 candidate 4 refine incumbent personal_ideal_alternation)

**Outcome:** predicts like existing model 'ideal_alternation_with_balance' (p_left RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of ideal_alternation_with_balance, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how often it switches. The single change is this shared H/T-balance penalty (the local-balance component of `local_representativeness`), added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which alternation alone cannot produce.

### run_hazard_switch_belief — pruned (experiment1 end of experiment)

**Outcome:** 121.2 nats behind individual_streak_aversion_lapse (5.3× dse)

**Hypothesis:** Refinement of `personal_switch_belief` (each person judges as more random the sequence that is more probable under their own subjective model of a coin as a process that switches between heads and tails with a personal probability). The one change: the believed coin has the gambler's fallacy built in — its chance of switching grows with the length of the current run (a shared rate of growth), on top of each person's own baseline switch belief — so a long run is judged far less probable than its number of switches alone implies. This addresses the critique that people penalise long runs beyond what the alternation rate captures.

### ideal_alternation_streak_penalty — pruned (experiment1 end of experiment)

**Outcome:** 78.0 nats behind individual_streak_aversion_lapse (5.1× dse)

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also notice the single longest streak of identical flips and treat a long streak as a sign of non-randomness, beyond what the overall alternation rate shows. The one change is a shared penalty on the length of the longest run (in flips), added because the critique shows that among irregular sequences people penalise long runs beyond what alternation rate captures, which the alternation-only model under-predicts.

### divisive_contrast_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 84.3 nats behind individual_streak_aversion_lapse (5.3× dse)

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the two sequences against each other by divisive contrast: how much less random one looks than the other is weighed relative to how non-random the pair looks overall. Each sequence's non-randomness is its departure from the person's own ideal switching rate plus its heads/tails imbalance, but a given difference decides the choice strongly when the partner is nearly ideal and only weakly when both sequences are clearly non-random, so the same sequence is judged differently beside a different partner.

### max_lopsided_stretch_ideal — pruned (experiment1 end of experiment)

**Outcome:** 118.7 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge a coin-flip sequence by its single most lopsided stretch: they notice the one contiguous run of flips in which heads most outnumber tails (or tails most outnumber heads), and a sequence looks random to the extent that this worst local excess matches what each person expects a random coin to produce. Each person carries their own ideal size for this worst stretch (relative to the sequence's length) — some expect even the most uneven stretch to be tiny, others tolerate a sizeable one — and the rest of the sequence, such as its overall alternation rate, matters only through that one most striking stretch.

### personal_lapse_ideal_alternation — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** People judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but the decision rule is not always engaged: on some trials a person does not compare the sequences at all and picks a side by guessing. How often this happens is a stable trait that differs between people, so some participants follow their alternation preference almost every trial while others are close to indifferent, which spreads individual choice proportions beyond what a single shared decisiveness produces.

### individual_balance_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 58.0 nats behind individual_streak_aversion_lapse (4.5× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random, but people differ in how much that imbalance matters to them — each person has their own weight on H/T balance, drawn from a population distribution, instead of everyone sharing one. The one change is making the balance penalty individual, because the critique shows participants differ in their preference for the more balanced sequence far more than a single shared balance weight allows.

### personal_balance_weight_alternation — rejected (experiment1 round 2 candidate 4 refine incumbent ideal_alternation_with_balance)

**Outcome:** predicts like existing model 'individual_balance_ideal_alternation' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_balance_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random — but how much a person cares about that balance is their own, drawn from a population distribution, rather than one weight shared by everyone. The one change is making the balance penalty participant-specific, because the critique shows people differ far more in how often they choose the more balanced sequence than a single shared balance weight allows.

### personal_span_glimpse_expectation — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'windowed_glimpse_expectation' (p_left RMSE 0.00193 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of windowed_glimpse_expectation, not a new hypothesis.

**Hypothesis:** Refinement of `windowed_glimpse_expectation`: people still read a sequence through a limited working-memory window and judge it random to the extent that the H/T mix of each glimpse matches what they expect from a random coin (given their own belief about how often a coin switches), but the width of that memory window now differs from person to person instead of being shared. The one change is a personal memory span: people with a narrow window judge mostly by flip-to-flip alternation, while people with a wide window see whole-sequence heads/tails balance and long runs, which predicts the large between-person spread in preferring the more balanced sequence that the critique of the incumbent reports.

### length_scaled_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 78.2 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still look less random, but people treat a departure from their ideal switching rate as evidence that grows with the number of flips it is observed over — the same off-ideal alternation rate is shrugged off in a short sequence and taken seriously in a long one. The one change is making alternation sensitivity grow with sequence length (as a power of the number of transitions, fitted), because the critique shows short sequences are judged by alternation less, relative to long ones, than the length-independent incumbent predicts.

### glimpse_evidence_accumulation — rejected (experiment1 round 2 candidate 5 refine chosen repair 1)

**Outcome:** MCMC did not converge (3000 divergent transitions of 12000; max R-hat 1.531 > 1.05; min bulk ESS 7 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of `windowed_glimpse_expectation`: people still read a sequence through a limited working-memory window and judge it random by how well the heads/tails mix of its glimpses matches what they expect from a random coin (given their own belief about how often a coin switches), but the evidence of non-randomness accumulates glimpse by glimpse instead of being averaged — every glimpse that looks too mixed or too lopsided adds to the impression, so a longer sequence, which offers more glimpses, is judged more decisively than a short one with the same average mismatch. The one change is summing the glimpse mismatches rather than averaging them, which predicts that short sequences are decided less sharply by their alternation than long ones, the direction the critique of the incumbent reports.

### tally_drift_ideal — pruned (experiment1 end of experiment)

**Outcome:** 186.0 nats behind individual_streak_aversion_lapse (5.9× dse)

**Hypothesis:** People keep a running tally of heads minus tails as they read a sequence, and judge it random to the extent that this tally wanders away from balance by about as much as they expect a fair coin's tally to wander: the whole path matters, so a streak that pushes the tally far off balance (even one later corrected) counts against a sequence, and so does a tally that never leaves balance at all. Each person has their own ideal amount of wandering — some expect the tally to hug zero, others tolerate large drifts — and picks the sequence whose tally drift is closer to that personal ideal.

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Outcome:** 287.9 nats behind individual_streak_aversion_lapse (10.4× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously designed coin sequences — a solid streak (HHHH…), strict alternation (HTHT…), pairs (HHTT…), triples (HHHTTT…) and two halves (HHHH TTTT) — and a sequence looks non-random to the extent that it closely resembles (differs in few flips from) any of these remembered patterns, with similarity falling off exponentially with the share of mismatched flips. People differ in whether their remembered set of "designed" sequences prominently includes strict alternation, so some find alternating sequences contrived and others find them random-looking, while streak-like resemblance is penalised by everyone.

### pattern_coverage_detection — pruned (experiment1 end of experiment)

**Outcome:** 194.8 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** People judge randomness by pattern detection: they scan a sequence for perceptible regularities — a streak of identical flips or a stretch of strict H/T alternation — and a sequence looks non-random in proportion to how much of it is covered by such a detected pattern. A stretch is noticed as a pattern once it is long enough, with streaks noticed at a shared length and alternating stretches at a length that differs from person to person, so the whole shape of the sequence (how many of its flips sit inside long runs or long alternations, not its overall switch rate) drives the choice.

### lapse_ideal_alternation_streak_aversion — rejected (experiment1 round 3 candidate 4 refine incumbent personal_lapse_ideal_alternation)

**Outcome:** predicts like existing model 'personal_lapse_ideal_alternation' (p_left RMSE 0.00073 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_lapse_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guess on a person-specific share of trials, but when they do compare the sequences they also notice the single longest streak of identical flips and treat a long streak (relative to the sequence's length) as a sign of non-randomness beyond what the alternation rate shows. The one change is this shared longest-streak penalty inside the engaged decision, added because the critique shows that among pairs matched on alternation rate people pick the sequence with the shorter longest run more often than the incumbent predicts.

### lapse_ideal_alternation_streak_penalty — rejected (experiment1 round 3 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'personal_lapse_ideal_alternation' (p_left RMSE 0.00056 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_lapse_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of `ideal_alternation_streak_penalty`: each person judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and everyone additionally treats a long longest streak of identical flips as a sign of non-randomness beyond what the alternation rate shows. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences and guesses, at a personal lapse rate drawn from a population, so the streak and alternation preferences are no longer diluted by trying to fit indifferent participants with one shared sensitivity — addressing the critique that people penalise long streaks among alternation-matched pairs more than the lapse-only incumbent predicts.

### local_window_ideal_alternation_lapse — pruned (experiment1 end of experiment)

**Outcome:** 119.8 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still judge a sequence as more random the closer its switching between heads and tails lies to their own personal ideal switching rate, and still guess on some trials at a personal, trait-like lapse rate, but they apply their ideal locally rather than to the sequence as a whole — they read the sequence a few flips at a time (a sliding stretch of three consecutive transitions) and each stretch is expected to switch at about the ideal rate, so a sequence's non-randomness is its average local departure from the ideal. The one change is this local (windowed) application of the ideal instead of the global alternation rate: two sequences with the same overall alternation rate differ when one bunches its repeats into a long streak and its switches into a rigid alternating stretch, which is exactly the streak aversion among alternation-matched pairs that the critique shows the incumbent under-predicts.

### glimpse_lapse_ideal_alternation — rejected (experiment1 round 3 candidate 4 refine incumbent personal_lapse_ideal_alternation repair 1)

**Outcome:** predicts like existing model 'local_window_ideal_alternation_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of local_window_ideal_alternation_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still compare a sequence's alternation with their own personal ideal switching rate and still guess on a person-specific share of trials, but they do not check the sequence's overall alternation rate. They read it through a short working-memory glimpse of about four flips and check each glimpse's switching rate against their ideal, so a sequence looks random to the extent that every local stretch alternates the way they expect. The one change is this local (glimpse-wise) evaluation of the same distance-from-ideal: a long streak next to a stretch that alternates heavily is judged non-random even when its overall alternation rate is near the ideal. That addresses the critique that among pairs matched on alternation rate people choose the sequence with the shorter longest run more often than the incumbent predicts.

### edge_salient_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 69.2 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge a sequence's randomness by how often it switches between heads and tails compared with their own personal ideal switching rate, but they do not weigh every transition equally: attention and memory favour the beginning and end of the sequence (a serial-position effect), so switches and repeats near the edges count more than those in the middle. This disagrees most sharply with the current best model on pairs matched in overall alternation rate and heads/tails balance that differ only in where their repeats sit — the best model predicts indifference, whereas this one predicts that a streak at the start or end of a sequence makes it look much less random than the same streak buried in the middle.

### serial_position_weighted_alternation — pruned (experiment1 end of experiment)

**Outcome:** 69.4 nats behind individual_streak_aversion_lapse (4.9× dse)

**Hypothesis:** People do not attend evenly to a coin-flip sequence: like a list read in order, its beginning and end are more salient than its middle (a serial-position effect), so the flip-to-flip switches and repeats at the edges of a sequence weigh more in their impression of how often it switches. Each person compares this salience-weighted switching rate with their own ideal switching rate for a random coin and picks the sequence that comes closer, so a streak sitting at the start or end of a sequence makes it look less random than the same streak buried in the middle.

### personal_streak_aversion_lapse_balance — rejected (experiment1 round 4 candidate 4 refine incumbent lapse_individual_balance_alternation)

**Outcome:** predicts like existing model 'individual_streak_aversion_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is a personal streak aversion: people also notice the longest streak of identical flips in a sequence and treat it as a sign of non-randomness, by an amount that differs from person to person (drawn from a population distribution) — addressing the critique that people avoid sequences with long (edge) streaks among alternation-matched pairs more than the incumbent predicts, and that individuals differ in their preference for the sequence with the shorter longest run more than its balance/alternation/lapse heterogeneity produces.

### falk_konold_dp — pruned (experiment1 end of experiment)

**Outcome:** 648.0 nats behind individual_streak_aversion_lapse (14.5× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

### motif_stack — pruned (experiment1 end of experiment)

**Outcome:** 485.2 nats behind individual_streak_aversion_lapse (11.9× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Outcome:** 816.9 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

### local_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 467.0 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

### indifference_band_streak_lapse — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** predicts like existing model 'individual_streak_aversion_lapse' (p_left RMSE 0.00195 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** People judge each sequence's randomness from how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is (each by weights of their own), and they guess on some trials at a personal lapse rate. The change is in the decision rule: when the two sequences seem about equally random, the difference falls inside an indifference band and the person effectively flips a coin. Only the part of the difference that goes beyond the band drives the choice, so near-ties look like guessing while clear differences are decided as sharply as before.

### palindrome_aversion_terminal_streak_lapse — rejected (experiment2 round 1 candidate 4 refine incumbent terminal_run_streak_aversion_lapse)

**Outcome:** predicts like existing model 'mirror_symmetry_streak_aversion_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of mirror_symmetry_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the streak the sequence ends on by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is that people also notice mirror symmetry: a sequence that reads the same forwards and backwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks designed rather than random, and so less random by a shared weight — addressing the critique that people choose the palindrome of a pair less often than the incumbent's alternation, balance and streak terms predict.

### run_length_variety_encoding — rejected (experiment2 round 3 candidate 1 lens 3)

**Outcome:** predicts like existing model 'run_length_variety_preference' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of run_length_variety_preference, not a new hypothesis.

**Hypothesis:** People encode a coin-flip sequence in working memory as a list of runs (stretches of identical flips) and judge how random it is partly by how varied that run list is: when every run has the same length (strict alternation HTHTHTHT, or pairs HHTTHHTT) the run list compresses to one repeated chunk size and the sequence looks rule-generated, whereas a mix of short and longer runs cannot be summarised by one chunk size and looks random. This run-length variety acts by a shared weight on top of the incumbent's judgement (each person's own ideal switching rate and sensitivity to departures from it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and on mirror symmetry, and a personal guessing rate) — addressing the critique (surviving FDR) that among pairs with the same alternation rate people choose the sequence with more distinct run lengths more often than the incumbent predicts.

### asymmetric_overalternation_mirror — rejected (experiment2 round 3 candidate 4 refine incumbent individual_alternation_sensitivity_mirror)

**Outcome:** predicts like existing model 'asymmetric_ideal_alternation_mirror' (p_left RMSE 0.00042 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of asymmetric_ideal_alternation_mirror, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate (with a personal sensitivity), still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that the penalty for missing one's ideal switching rate is asymmetric: switching more often than the ideal (over-alternation) costs a sequence a different amount, by a shared fitted factor, than switching less often (streakiness), because streaks look like a broken coin while extra alternation still looks "mixed" — addressing the critique (surviving FDR) that among two highly alternating sequences people choose the more alternating one more often than the symmetric quadratic ideal-rate term predicts.
