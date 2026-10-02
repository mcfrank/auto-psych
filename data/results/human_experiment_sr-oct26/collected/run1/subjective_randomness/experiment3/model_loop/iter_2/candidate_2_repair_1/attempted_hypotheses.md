# Tried before

74 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### lapse_individual_balance_alternation — pruned (experiment2 end of experiment)

**Outcome:** 211.4 nats behind most_lopsided_stretch_aversion (6.7× dse)

**Hypothesis:** Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.

### edge_salient_ideal_alternation — pruned (experiment1 end of experiment)

**Outcome:** 69.2 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge a sequence's randomness by how often it switches between heads and tails compared with their own personal ideal switching rate, but they do not weigh every transition equally: attention and memory favour the beginning and end of the sequence (a serial-position effect), so switches and repeats near the edges count more than those in the middle. This disagrees most sharply with the current best model on pairs matched in overall alternation rate and heads/tails balance that differ only in where their repeats sit — the best model predicts indifference, whereas this one predicts that a streak at the start or end of a sequence makes it look much less random than the same streak buried in the middle.

### serial_position_weighted_alternation — pruned (experiment1 end of experiment)

**Outcome:** 69.4 nats behind individual_streak_aversion_lapse (4.9× dse)

**Hypothesis:** People do not attend evenly to a coin-flip sequence: like a list read in order, its beginning and end are more salient than its middle (a serial-position effect), so the flip-to-flip switches and repeats at the edges of a sequence weigh more in their impression of how often it switches. Each person compares this salience-weighted switching rate with their own ideal switching rate for a random coin and picks the sequence that comes closer, so a streak sitting at the start or end of a sequence makes it look less random than the same streak buried in the middle.

### edge_weighted_alternation_memory — pruned (experiment2 end of experiment)

**Outcome:** 350.6 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People encode a coin-flip sequence with serial-position effects in memory: the flips at the start and end of the sequence (primacy and recency) are encoded more strongly than those in the middle, so the switching pattern they perceive is dominated by what happens at the sequence's edges. Each person compares this edge-weighted impression of how often the sequence switches between heads and tails with their own personal ideal switching rate, and on some trials guesses at a personal lapse rate; a streak or rigid alternation sitting at an edge therefore makes a sequence look much less random than the same pattern buried in the middle.

### individual_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Outcome:** 135.1 nats behind most_lopsided_stretch_aversion (5.3× dse)

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that people also notice the longest streak of identical flips and treat it as a sign of non-randomness by a weight that is each person's own (drawn from a population distribution, so some people are strongly streak-averse and others barely care), addressing the critique that individuals differ in preferring the sequence with the shorter longest run more than the incumbent produces, and that streaks are penalised beyond its alternation and balance terms.

### personal_streak_aversion_lapse_balance — rejected (experiment1 round 4 candidate 4 refine incumbent lapse_individual_balance_alternation)

**Outcome:** predicts like existing model 'individual_streak_aversion_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is a personal streak aversion: people also notice the longest streak of identical flips in a sequence and treat it as a sign of non-randomness, by an amount that differs from person to person (drawn from a population distribution) — addressing the critique that people avoid sequences with long (edge) streaks among alternation-matched pairs more than the incumbent predicts, and that individuals differ in their preference for the sequence with the shorter longest run more than its balance/alternation/lapse heterogeneity produces.

### lapse_individual_streak_aversion — pruned (experiment2 end of experiment)

**Outcome:** 195.6 nats behind most_lopsided_stretch_aversion (6.9× dse)

**Hypothesis:** Refinement of `personal_lapse_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guesses on some trials at a personal, trait-like lapse rate, but people also treat a long streak of identical flips as a sign of non-randomness, and how strongly a streak puts them off is each person's own trait (a personal weight on the longest run relative to sequence length, drawn from a population). The one change is this individual streak-aversion weight, added because the critique shows participants differ in their preference for the sequence with the shorter longest run more than the model's alternation, balance and lapse heterogeneity produces, and that streaks are penalised beyond alternation among alternation-matched pairs.

### edge_streak_balance_alternation_lapse — pruned (experiment2 end of experiment)

**Outcome:** 154.2 nats behind most_lopsided_stretch_aversion (5.8× dse)

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that a streak sitting at the very start or end of a sequence is especially salient (primacy/recency of what is read first and last): the number of flips by which a run of identical flips at either edge exceeds two makes the sequence look less random by a shared weight, while the same streak buried in the middle carries no extra penalty — addressing the critique that among alternation-matched pairs people avoid the sequence with an edge streak more than the incumbent predicts.

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

### leaky_memory_bayesian_randomness — pruned (experiment2 end of experiment)

**Outcome:** 948.4 nats behind most_lopsided_stretch_aversion (10.5× dse)

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent a fair coin explains it better than the non-random generators one might suspect (a biased coin, or a sticky/switchy coin whose switching probability is unknown), with the marginal likelihood of each alternative computed from the sequence's head/tail and repeat/switch counts. The one distortion is leaky memory: as they read left to right, earlier flips fade, so the evidence each flip contributes is discounted geometrically with its distance from the end of the sequence, which makes a streak or rigid pattern at the end of a sequence count as much stronger evidence of non-randomness than the same pattern at its start (people also guess on some trials at a personal rate).

### lempel_ziv_incompressibility — pruned (experiment2 end of experiment)

**Outcome:** 637.7 nats behind most_lopsided_stretch_aversion (8.2× dse)

**Hypothesis:** People judge a coin-flip sequence's randomness by how hard it is to describe compactly: reading it left to right, they notice whenever the next stretch of flips merely copies something already seen, and a sequence looks random in proportion to how many genuinely new chunks it takes to spell it out (its Lempel-Ziv compression complexity). How strongly this incompressibility drives the choice is each person's own trait, drawn from a population, so streaks, rigid alternation and repeated motifs all count against a sequence only through the single fact that they make it compressible.

### shared_stretch_cancellation — pruned (experiment2 end of experiment)

**Outcome:** 277.2 nats behind most_lopsided_stretch_aversion (6.8× dse)

**Hypothesis:** People compare the two sequences side by side and cancel what they share: flips that both sequences have in common at their start and at their end are discounted as uninformative, and each sequence's randomness is judged only on the stretch where the two differ (plus the flip on either side of it, so its switches into and out of that stretch count). On that differing stretch each person still judges randomness by how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is, and guesses on some trials; so the same sequence is judged differently beside a partner that shares its opening or its ending than beside one that shares nothing.

### terminal_run_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Outcome:** 123.2 nats behind most_lopsided_stretch_aversion (5.7× dse)

**Hypothesis:** Refinement of the incumbent `individual_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that the streak a sequence ends on is weighed differently from streaks elsewhere: the run of identical flips at the end of a sequence (the last thing read) shifts its apparent randomness by a shared weight of its own, which may be lenient or harsh, addressing the critique that people choose the sequence with the longer final run more often than the incumbent's position-blind streak term predicts.

### terminal_discounted_streak_lapse — pruned (experiment2 end of experiment)

**Outcome:** 125.9 nats behind most_lopsided_stretch_aversion (6.6× dse)

**Hypothesis:** Refinement of the incumbent `individual_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that the streak a sequence ends on is weighed differently from streaks that have already been closed off: because sequences are read left to right, a run still in progress at the last flip is judged as an unfinished streak, so its length counts by a fitted shared factor (discounted or amplified) when deciding which streak is the sequence's most striking one — addressing the critique that people choose the sequence with the longer final run more often than the position-blind longest-run term predicts.

### primacy_recency_streak_balance_lapse — pruned (experiment2 end of experiment)

**Outcome:** 184.0 nats behind most_lopsided_stretch_aversion (7.0× dse)

**Hypothesis:** Refinement of `edge_streak_balance_alternation_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate, and streaks at the edges of a sequence still carry an extra penalty. The one change is that the opening streak and the closing streak are no longer treated as one interchangeable "edge" streak: the streak a person reads last (recency) and the streak they read first (primacy) each make the sequence look less random by a weight of its own, so a run at the end can count for more than the same run at the start — addressing the critique that people penalise a long final run more than the incumbent's position-blind longest-run term predicts.

### worst_stretch_switch_deficit — pruned (experiment2 end of experiment)

**Outcome:** 293.2 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge a coin-flip sequence by its single most un-random-looking stretch: they scan every contiguous stretch of flips and count how far its number of heads/tails switches falls short of (a streak) or exceeds (rigid alternation) the number they expect from their own ideal switching rate, and the sequence is only as random as its worst stretch. Because the shortfall is counted in switches rather than as a rate, a long stretch that is modestly off can outweigh a short one that is badly off, so a sequence's randomness is decided by one striking stretch rather than by its overall alternation rate, balance or longest run (people also guess on some trials at a personal rate).

### indifference_band_streak_lapse — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** predicts like existing model 'individual_streak_aversion_lapse' (p_left RMSE 0.00195 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** People judge each sequence's randomness from how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is (each by weights of their own), and they guess on some trials at a personal lapse rate. The change is in the decision rule: when the two sequences seem about equally random, the difference falls inside an indifference band and the person effectively flips a coin. Only the part of the difference that goes beyond the band drives the choice, so near-ties look like guessing while clear differences are decided as sharply as before.

### running_lead_surprise_path — pruned (experiment2 end of experiment)

**Outcome:** 467.7 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People keep a running tally of heads minus tails as they read a sequence left to right, and at every flip they ask how surprising the current lead is for a fair coin given how many flips they have seen so far (a lead of two after two flips is far more striking than a lead of two after eight). A sequence's impression of randomness is the average of this moment-by-moment surprise along the whole path, compared with each person's own ideal level of surprise (some expect the tally to hug balance, others expect it to wander), so an opening streak — which makes the early tally lopsided when few flips have been seen — counts against a sequence far more than the same streak at the end; on some trials a person guesses at a personal lapse rate.

### mirror_symmetry_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Outcome:** 78.1 nats behind most_lopsided_stretch_aversion (6.1× dse)

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is that people also notice mirror symmetry: a sequence that reads the same forwards and backwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks deliberately designed and so less random, by a shared weight — addressing the critique (surviving FDR) that people choose the palindrome of a pair less often than the incumbent's alternation, balance and streak terms predict.

### palindrome_aversion_terminal_streak_lapse — rejected (experiment2 round 1 candidate 4 refine incumbent terminal_run_streak_aversion_lapse)

**Outcome:** predicts like existing model 'mirror_symmetry_streak_aversion_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of mirror_symmetry_streak_aversion_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the streak the sequence ends on by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is that people also notice mirror symmetry: a sequence that reads the same forwards and backwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks designed rather than random, and so less random by a shared weight — addressing the critique that people choose the palindrome of a pair less often than the incumbent's alternation, balance and streak terms predict.

### mirror_symmetry_terminal_streak_lapse — pruned (experiment2 end of experiment)

**Outcome:** 82.7 nats behind most_lopsided_stretch_aversion (5.6× dse)

**Hypothesis:** Refinement of `terminal_discounted_streak_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the most striking streak (with the run still in progress at the end weighed by a shared factor) by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that people notice mirror symmetry: a sequence of four or more flips that reads the same backwards as forwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks designed, and so less random, by a shared weight — addressing the critique that people choose the palindrome of a pair less often than the alternation, balance and streak terms predict.

### side_bias_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Outcome:** 135.6 nats behind most_lopsided_stretch_aversion (5.3× dse)

**Hypothesis:** People judge each sequence's randomness from how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is (each by weights of their own), and they guess on some trials at a personal lapse rate. The change is in the decision rule: each person also has a habitual leaning toward one response button (left or right), a position bias drawn from a population that may lean left overall, which tips every comparison toward that side by a fixed amount — so the bias matters most when the two sequences look about equally random, and a person's overall left-choice rate departs from one half regardless of which sequences they see.

### opening_run_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Outcome:** 101.2 nats behind most_lopsided_stretch_aversion (6.2× dse)

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the streak the sequence ends on by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is primacy: the run of identical flips a sequence opens with (the first thing read, which sets the first impression) also shifts its apparent randomness by a shared weight of its own, separate from the closing run's weight — addressing the critique (surviving FDR) that people choose the sequence with the longer opening run less often than the incumbent, which weighs the final run but not the first, predicts.

### exemplar_random_vs_designed_gcm — pruned (experiment2 end of experiment)

**Outcome:** 531.0 nats behind most_lopsided_stretch_aversion (9.0× dse)

**Hypothesis:** People judge randomness by exemplar categorisation: they hold two remembered sets of same-length coin sequences, "random-looking" exemplars (balanced head/tail counts with no streak longer than two) and "designed" exemplars (a solid streak, strict alternation, repeated pairs HHTT…, repeated triples HHHTTT…, and two halves HHHHTTTT), and a sequence looks random to the extent its summed similarity to the random exemplars outweighs its summed similarity to the designed ones, with similarity falling off exponentially with the share of flips that differ. The sequence with the stronger random-versus-designed evidence is chosen, by a decisiveness that differs from person to person, and people guess on some trials at a personal lapse rate.

### short_pattern_evenness_lapse — pruned (experiment2 end of experiment)

**Outcome:** 446.4 nats behind most_lopsided_stretch_aversion (7.2× dse)

**Hypothesis:** People judge randomness by pattern evenness: they expect a random coin to produce every short pattern about equally often — heads and tails, each of the four flip pairs (HH, HT, TH, TT) and each of the flip triples — so a sequence looks non-random to the extent that a few short patterns dominate it (as in a streak, which is all HH, or strict alternation, which is all HT and TH), measured as how far the spread of its overlapping one-, two- and three-flip patterns falls short of the most even spread its length allows. How strongly this pattern unevenness puts a person off is their own trait, drawn from a population, and on some trials people guess at a personal lapse rate.

### periodic_chunk_repetition_aversion — pruned (experiment2 end of experiment)

**Outcome:** 79.7 nats behind most_lopsided_stretch_aversion (6.4× dse)

**Hypothesis:** People see a sequence that is built by repeating a short chunk of three or four flips (translational repetition, such as HTTHTTHT or HTTHHTTH) as designed rather than random, so it looks less random by a shared weight on top of the incumbent's judgement (each person's ideal switching rate, personal weights on heads/tails imbalance and on the longest streak, a shared weight on the final streak and on mirror symmetry, and a personal guessing rate). This disagrees most sharply with the current best model on pairs where one sequence repeats a short chunk while having a near-ideal switching rate, good balance and no long streak (e.g. HTHHTHHT or HTTHTTHT against a streakier but non-repeating sequence): the best model calls these sequences highly random, while this model predicts people reject them.

### individual_alternation_sensitivity_mirror — pruned (experiment2 end of experiment)

**Outcome:** 45.7 nats behind most_lopsided_stretch_aversion (4.1× dse)

**Hypothesis:** Refinement of the incumbent `mirror_symmetry_streak_aversion_lapse`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry (palindromes look designed) by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that how sharply a departure from one's ideal switching rate counts against a sequence is each person's own trait (drawn from a population) rather than shared: some people are strict, consistent judges of alternation and others barely discriminate — addressing the critique that people differ in how consistently they agree with the majority choice more than the incumbent's person-level weights and lapse rates produce.

### personal_sensitivity_opening_run_lapse — pruned (experiment2 end of experiment)

**Outcome:** 58.1 nats behind most_lopsided_stretch_aversion (4.5× dse)

**Hypothesis:** Refinement of `opening_run_streak_aversion_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the closing run and the opening run of identical flips by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that how sharply a person's choices follow the distance from their ideal switching rate is their own trait (a personal alternation sensitivity drawn from a population, instead of one shared sensitivity), so some engaged people decide almost deterministically by alternation while others barely discriminate — addressing the critique that people differ in how consistently they agree with the majority choice more than the model's person-level ideals, weights and lapse rates produce.

### run_length_variety_encoding — rejected (experiment2 round 3 candidate 1 lens 3)

**Outcome:** predicts like existing model 'run_length_variety_preference' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of run_length_variety_preference, not a new hypothesis.

**Hypothesis:** People encode a coin-flip sequence in working memory as a list of runs (stretches of identical flips) and judge how random it is partly by how varied that run list is: when every run has the same length (strict alternation HTHTHTHT, or pairs HHTTHHTT) the run list compresses to one repeated chunk size and the sequence looks rule-generated, whereas a mix of short and longer runs cannot be summarised by one chunk size and looks random. This run-length variety acts by a shared weight on top of the incumbent's judgement (each person's own ideal switching rate and sensitivity to departures from it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and on mirror symmetry, and a personal guessing rate) — addressing the critique (surviving FDR) that among pairs with the same alternation rate people choose the sequence with more distinct run lengths more often than the incumbent predicts.

### run_rhythm_evidence_personal_coin — pruned (experiment2 end of experiment)

**Outcome:** 332.1 nats behind most_lopsided_stretch_aversion (8.9× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent that a coin explains it better than a "regular-rhythm" generator that builds sequences from runs of a few recurring lengths (so sequences whose runs all have the same length, such as HTHTHTHT, HHTTHHTT or one solid streak, look designed, and sequences with a variety of run lengths look random). The one distortion is that each person's mental model of the fair coin switches between heads and tails at their own personal rate rather than exactly half the time, so people differ in how much alternation they expect from a random coin while all still prize irregular run structure.

### asymmetric_overalternation_mirror — rejected (experiment2 round 3 candidate 4 refine incumbent individual_alternation_sensitivity_mirror)

**Outcome:** predicts like existing model 'asymmetric_ideal_alternation_mirror' (p_left RMSE 0.00042 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of asymmetric_ideal_alternation_mirror, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate (with a personal sensitivity), still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that the penalty for missing one's ideal switching rate is asymmetric: switching more often than the ideal (over-alternation) costs a sequence a different amount, by a shared fitted factor, than switching less often (streakiness), because streaks look like a broken coin while extra alternation still looks "mixed" — addressing the critique (surviving FDR) that among two highly alternating sequences people choose the more alternating one more often than the symmetric quadratic ideal-rate term predicts.

### asymmetric_ideal_rate_decisiveness — pruned (experiment2 end of experiment)

**Outcome:** 55.8 nats behind most_lopsided_stretch_aversion (4.0× dse)

**Hypothesis:** Refinement of `personal_decisiveness_mirror_streak_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, still acts on their impression with a personal decisiveness, and still guesses on some trials at a personal lapse rate. The one change is that departures from one's ideal switching rate are judged asymmetrically: switching too often (over-alternation) counts against a sequence by a different, shared fraction of what switching too rarely (streakiness) does, rather than both directions counting equally — addressing the critique (surviving FDR) that among highly alternating pairs people choose the more alternating sequence far more often than a symmetric ideal-rate penalty implies.

### length_load_encoding_noise — pruned (experiment2 end of experiment)

**Outcome:** 44.8 nats behind most_lopsided_stretch_aversion (4.1× dse)

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror` by a working-memory load process: to compare two sequences people must hold both in memory, and every additional flip adds encoding noise, so the same difference in how random the two look decides the choice sharply for short sequences and only weakly for long ones (the noise grows as a power of the number of flips, by a shared fitted exponent that could also come out negative if longer sequences give more evidence). How random each sequence looks is still the incumbent's judgement — each person's own ideal switching rate and sensitivity to it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and mirror symmetry — and people still guess on some trials at a personal lapse rate.

### linear_distance_ideal_alternation_mirror — pruned (experiment2 end of experiment)

**Outcome:** 61.4 nats behind most_lopsided_stretch_aversion (5.1× dse)

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate (with their own sensitivity), still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is the shape of the ideal-rate penalty: people count how many switches a sequence is off from their ideal and each missing or surplus switch costs the same amount (a penalty proportional to the distance from the ideal, not to its square), so small departures near the ideal are noticed and acted on rather than shrugged off, and large departures are not punished disproportionately — addressing the critique (surviving FDR) that among two highly alternating sequences people choose the more alternating one more often than the incumbent's quadratic ideal-rate penalty implies.

### gist_typicality_class_size — pruned (experiment2 end of experiment)

**Outcome:** 894.9 nats behind most_lopsided_stretch_aversion (12.0× dse)

**Hypothesis:** People judge randomness by typicality of a sequence's gist: they register only two summary facts — how many heads it has and how many runs (stretches of identical flips) it breaks into — and a sequence looks random to the extent that many other sequences of the same length share that same gist, so a gist that a fair coin could produce in many ways (moderate balance, moderate switching) looks random while a rare gist (one solid streak, strict alternation, all heads then all tails) looks designed. The sequence whose gist is more common is chosen, with a decisiveness that differs from person to person, and people guess on some trials at a personal lapse rate.

### first_read_anchor_weighting — rejected (experiment2 round 4 candidate 1 lens 6)

**Outcome:** predicts like existing model 'run_length_variety_preference' (p_left RMSE 0.00152 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of run_length_variety_preference, not a new hypothesis.

**Hypothesis:** People do not judge the two sequences independently: they read the left sequence first and it becomes the anchor, so its impression of randomness (good or bad) is weighed more heavily than the one formed for the second sequence, which is only judged as an adjustment from that anchor. As a result the choice depends not only on which sequence looks more random but on how random the pair looks overall — when both look clearly non-random the first-read sequence's flaws loom larger (or smaller, by a shared fitted anchoring weight), so the same sequence is chosen at different rates beside different partners; each sequence's randomness itself is judged as in the current best model (personal ideal switching rate and sensitivity, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate).

### asymmetric_alternation_run_variety — rejected (experiment2 round 4 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'run_variety_asymmetric_ideal_rate' (p_left RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of run_variety_asymmetric_ideal_rate, not a new hypothesis.

**Hypothesis:** Refinement of `asymmetric_ideal_alternation_mirror` (rank 1): each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, with their own sensitivity and with switching too often penalised by a different, shared fraction of the penalty for switching too rarely; still penalises lopsided heads/tails counts and the longest streak by weights of their own; still weighs the final streak and mirror symmetry by shared weights; and still guesses on some trials at a personal lapse rate. The one change is grafting in the incumbent's run-length variety: a sequence whose runs of identical flips all share one length (HTHTHTHT, HHTTHHTT) looks rhythmic and designed, while one mixing runs of different lengths looks random, by a shared weight on the number of distinct run lengths relative to the most its length allows — so the leniency toward over-alternation that the critique (high-alternation pairs: people choose the more alternating sequence more often than predicted) asks for is estimated separately from the rhythm penalty that strictly alternating sequences also carry.

### asymmetric_ideal_lopsided_stretch — rejected (experiment3 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'asymmetric_tolerance_lopsided_stretch' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of asymmetric_tolerance_lopsided_stretch, not a new hypothesis.

**Hypothesis:** Refinement of `run_variety_asymmetric_ideal_rate` (rank 1, statistically tied with the best): people still compare a sequence's switching rate with their own ideal (with their own sensitivity), judging switching too often more leniently than switching too rarely by a shared fraction, still prefer varied run lengths, penalise heads/tails imbalance and the longest streak by personal weights, weigh the final streak and mirror symmetry by shared weights, and guess at a personal lapse rate. The one change is grafting in the incumbent's most lopsided stretch: the single contiguous stretch in which heads most outnumber tails (or vice versa), even when not one unbroken streak, makes a sequence look less random by a shared weight — so that local clustering of repeats is penalised as a striking excess while over-alternation is forgiven by the asymmetric ideal, predicting that among highly alternating pairs people pick the more alternating sequence even more often than either parent model alone predicts.

### personal_overalternation_leniency — rejected (experiment3 round 1 candidate 4 refine incumbent asymmetric_tolerance_lopsided_stretch)

**Outcome:** predicts like existing model 'asymmetric_tolerance_lopsided_stretch' (p_left RMSE 0.00180 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of asymmetric_tolerance_lopsided_stretch, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `asymmetric_tolerance_lopsided_stretch`: people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, and still judge it by their own ideal switching rate and sensitivity, personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate, with switching more often than the ideal costing less than switching too rarely. The one change is that this leniency toward over-alternation is each person's own trait (drawn from a population) rather than shared: some people treat a sequence that switches more than they expect as nearly as random as an ideal one, while others punish over-alternation as hard as streakiness — predicting that people disagree most on pairs of highly alternating sequences, beyond what their differing ideal rates produce.

### personal_over_alternation_leniency — rejected (experiment3 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'asymmetric_tolerance_lopsided_stretch' (p_left RMSE 0.00180 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of asymmetric_tolerance_lopsided_stretch, not a new hypothesis.

**Hypothesis:** Refinement of `most_lopsided_stretch_aversion` (rank 3): people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, and still judge it by their own ideal switching rate and sensitivity, personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is that departures from one's ideal switching rate count differently in the two directions, and how lenient a person is toward switching more often than their ideal (relative to the cost of switching too rarely) is their own trait drawn from a population — some people barely mind over-alternation while others penalise it as much as streakiness — so people disagree about pairs of highly alternating sequences more than a single shared leniency (the incumbent's) allows.

### bayesian_generator_personal_suspicion — rejected (experiment3 round 2 candidate 1 lens 4)

**Outcome:** MCMC did not converge (max R-hat 1.234 > 1.05; min bulk ESS 13 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge randomness normatively, by Bayesian inference over generators: for each sequence they work out how probable it is that it came from a fair coin rather than from a non-random generator (a biased coin, a sticky-or-switchy coin, or a mirror- or complement-symmetric construction), and pick the sequence with the higher posterior probability of being random. The one distortion is a personal prior suspicion: each person brings their own prior belief that a sequence was made by a non-random generator, so a suspicious person's posteriors for most sequences are near zero and they discriminate only among the most random-looking ones, while a trusting person's are near one and they discriminate only among the most regular-looking ones — which makes people differ in how consistently they agree with the majority depending on which pairs they see.
