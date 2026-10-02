# Tried before

9 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_switch_rate_prototype — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_ideal_alternation' (p_left RMSE 0.00008 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_ideal_alternation, not a new hypothesis.

**Hypothesis:** Each person carries their own prototype of how often a genuinely random coin switches between heads and tails, and judges a sequence as random to the extent its switch rate is close to that personal prototype. People differ substantially in this prototype (some expect heavy alternation, others expect near-independent switching), so the same pair can be judged in opposite directions by different participants.

### person_prototype_representativeness — rejected (experiment1 round 0 candidate 4 refine incumbent local_representativeness)

**Outcome:** predicts like existing model 'individual_alternation_prototype' (p_left RMSE 0.00063 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_alternation_prototype, not a new hypothesis.

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge a sequence random when it is locally balanced and irregular, but the alternation rate each person's mental prototype of randomness expects differs between people — some expect strong over-alternation, others expect streakier sequences — instead of one shared over-alternating prototype. The single change is that the prototype alternation rate becomes a per-participant parameter (free to lie below or above 0.5) drawn from a population distribution, addressing the critique that real individual differences in alternation preference are far larger than the single-population model produces.

### balanced_personal_ideal_alternation — rejected (experiment1 round 1 candidate 4 refine incumbent personal_ideal_alternation)

**Outcome:** predicts like existing model 'ideal_alternation_with_balance' (p_left RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of ideal_alternation_with_balance, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how often it switches. The single change is this shared H/T-balance penalty (the local-balance component of `local_representativeness`), added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which alternation alone cannot produce.

### personal_balance_weight_alternation — rejected (experiment1 round 2 candidate 4 refine incumbent ideal_alternation_with_balance)

**Outcome:** predicts like existing model 'individual_balance_ideal_alternation' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_balance_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random — but how much a person cares about that balance is their own, drawn from a population distribution, rather than one weight shared by everyone. The one change is making the balance penalty participant-specific, because the critique shows people differ far more in how often they choose the more balanced sequence than a single shared balance weight allows.

### personal_span_glimpse_expectation — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'windowed_glimpse_expectation' (p_left RMSE 0.00193 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of windowed_glimpse_expectation, not a new hypothesis.

**Hypothesis:** Refinement of `windowed_glimpse_expectation`: people still read a sequence through a limited working-memory window and judge it random to the extent that the H/T mix of each glimpse matches what they expect from a random coin (given their own belief about how often a coin switches), but the width of that memory window now differs from person to person instead of being shared. The one change is a personal memory span: people with a narrow window judge mostly by flip-to-flip alternation, while people with a wide window see whole-sequence heads/tails balance and long runs, which predicts the large between-person spread in preferring the more balanced sequence that the critique of the incumbent reports.

### glimpse_evidence_accumulation — rejected (experiment1 round 2 candidate 5 refine chosen repair 1)

**Outcome:** MCMC did not converge (3000 divergent transitions of 12000; max R-hat 1.531 > 1.05; min bulk ESS 7 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of `windowed_glimpse_expectation`: people still read a sequence through a limited working-memory window and judge it random by how well the heads/tails mix of its glimpses matches what they expect from a random coin (given their own belief about how often a coin switches), but the evidence of non-randomness accumulates glimpse by glimpse instead of being averaged — every glimpse that looks too mixed or too lopsided adds to the impression, so a longer sequence, which offers more glimpses, is judged more decisively than a short one with the same average mismatch. The one change is summing the glimpse mismatches rather than averaging them, which predicts that short sequences are decided less sharply by their alternation than long ones, the direction the critique of the incumbent reports.

### lapse_ideal_alternation_streak_aversion — rejected (experiment1 round 3 candidate 4 refine incumbent personal_lapse_ideal_alternation)

**Outcome:** predicts like existing model 'personal_lapse_ideal_alternation' (p_left RMSE 0.00073 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_lapse_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guess on a person-specific share of trials, but when they do compare the sequences they also notice the single longest streak of identical flips and treat a long streak (relative to the sequence's length) as a sign of non-randomness beyond what the alternation rate shows. The one change is this shared longest-streak penalty inside the engaged decision, added because the critique shows that among pairs matched on alternation rate people pick the sequence with the shorter longest run more often than the incumbent predicts.

### lapse_ideal_alternation_streak_penalty — rejected (experiment1 round 3 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'personal_lapse_ideal_alternation' (p_left RMSE 0.00056 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_lapse_ideal_alternation, not a new hypothesis.

**Hypothesis:** Refinement of `ideal_alternation_streak_penalty`: each person judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and everyone additionally treats a long longest streak of identical flips as a sign of non-randomness beyond what the alternation rate shows. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences and guesses, at a personal lapse rate drawn from a population, so the streak and alternation preferences are no longer diluted by trying to fit indifferent participants with one shared sensitivity — addressing the critique that people penalise long streaks among alternation-matched pairs more than the lapse-only incumbent predicts.

### glimpse_lapse_ideal_alternation — rejected (experiment1 round 3 candidate 4 refine incumbent personal_lapse_ideal_alternation repair 1)

**Outcome:** predicts like existing model 'local_window_ideal_alternation_lapse' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of local_window_ideal_alternation_lapse, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still compare a sequence's alternation with their own personal ideal switching rate and still guess on a person-specific share of trials, but they do not check the sequence's overall alternation rate. They read it through a short working-memory glimpse of about four flips and check each glimpse's switching rate against their ideal, so a sequence looks random to the extent that every local stretch alternates the way they expect. The one change is this local (glimpse-wise) evaluation of the same distance-from-ideal: a long streak next to a stretch that alternates heavily is judged non-random even when its overall alternation rate is near the ideal. That addresses the critique that among pairs matched on alternation rate people choose the sequence with the shorter longest run more often than the incumbent predicts.
