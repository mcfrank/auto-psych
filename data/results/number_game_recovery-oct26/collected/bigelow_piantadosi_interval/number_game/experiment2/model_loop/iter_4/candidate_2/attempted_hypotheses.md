# Tried before

41 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### shared_property_match_fraction — pruned (experiment1 end of experiment)

**Outcome:** 353.2 nats behind criterion_shifted_redundant_examples (9.4× dse)

**Hypothesis:** People do not weigh competing concepts or the size of each concept; instead they note every property that all the examples have in common (parity, final digit, tens decade, being a multiple, power, square, cube or prime, lying within the examples' range) and endorse a target in proportion to how many of those shared properties it also has. The more of the examples' common properties the target matches, the more likely a "yes", regardless of how many numbers in 1-100 share those properties.

### log_magnitude_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 612.2 nats behind criterion_shifted_redundant_examples (15.2× dse)

**Hypothesis:** People ignore arithmetic rules entirely and generalize by exemplar similarity on a compressed (logarithmic) mental number line: the chance of saying "yes" falls off smoothly with the target's distance to the nearest example, and the falloff is wider when the examples themselves are spread far apart. This disagrees most sharply with the rule-based best model on rule-consistent targets far from every example (e.g. 64 after 16,32,2,8 — rules say yes, similarity says unlikely) and on rule-violating targets nestled among the examples (e.g. 20 after 16,32,2,8 — rules say no, similarity says likely).

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 354.9 nats behind criterion_shifted_redundant_examples (12.3× dse)

**Hypothesis:** People do not entertain candidate rules at all; instead they judge a target by its featural similarity to each example they saw, treating numbers as bundles of salient properties (parity, divisibility by 3, 4, 5 and 10, last digit, decade, being prime, square or a power of two). Each example's similarity falls off exponentially with the number of properties on which it and the target disagree, and the chance of saying "yes" rises with the average similarity across the examples (an exemplar, GCM-style judgment with no size principle).

### concept_learning_with_guessing — pruned (experiment1 end of experiment)

**Outcome:** 53.1 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over 42 weighted rules and all intervals (size principle, 10% noisy examples), but on a fraction of trials they do not consult the inferred concept at all and instead guess, answering "yes" with their own guessing bias. The one change is this inferred guessing (lapse) rate with a yes-bias, which puts a floor under "yes" for targets far outside every plausible concept (the far-outside-range yeses the incumbent under-produces) and a ceiling below 1 for the examples themselves.

### interval_width_power_prior — pruned (experiment1 end of experiment)

**Outcome:** 60.5 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 42 weighted rules and all intervals of 1-100 with the size principle, but their prior over intervals is not flat — it rises or falls as a power of the interval's width, with the exponent inferred. A prior that favours wide intervals offsets the size principle and makes people generalize further beyond the range of the examples (the incumbent under-produces "yes" to targets far outside that range); an exponent of zero recovers the incumbent.

### tempered_size_principle_rules_intervals — pruned (experiment1 end of experiment)

**Outcome:** 216.6 nats behind criterion_shifted_redundant_examples (6.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over mathematical rules and intervals as in that model, but they discount the evidence of the examples — each example counts for only a fraction (or a multiple) of a fully independent random draw from the concept, by a fitted evidence weight — so the size principle is tempered and generalization stays broader than strong sampling predicts, reaching targets well outside the examples' range. The single change is this fitted tempering of the likelihood (the strength of the size principle), which the original fixes at full strength.

### serial_position_weighted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 53.6 nats behind criterion_shifted_redundant_examples (3.2× dse)

**Hypothesis:** People read the examples one at a time and integrate them into the same Bayesian concept learning over rules and intervals (with occasional guessing) as the best model, but their memory and attention are unequal across the sequence: each example's evidence is weighted by its serial position, with a single inferred gradient that favours the most recent examples (leaky memory / forgetting) or the earliest ones (primacy, fading attention). Because earlier and later examples then count differently, generalization from multi-example sets depends on the order in which the same numbers were shown, unlike the order-blind incumbent; a flat gradient recovers it.

### inferred_example_noise_belief — pruned (experiment1 end of experiment)

**Outcome:** 50.9 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** People do Bayesian concept learning over weighted mathematical rules and number intervals with the size principle, but they distort one assumption of the normative learner: how often they believe an example is a random number not produced by the concept. That believed noise rate is a free, inferred quantity rather than a fixed small value; the more examples people discount as noise, the more readily they accept rules that fit only some of the examples (e.g. "ends in 2" after 92,82,27,42) and the broader the intervals they entertain.

### weak_sampling_concept_elimination — pruned (experiment1 end of experiment)

**Outcome:** 95.3 nats behind criterion_shifted_redundant_examples (5.5× dse)

**Hypothesis:** People generalize by weak-sampling hypothesis elimination: they entertain the same kinds of candidate concepts (mathematical rules and numeric intervals) but treat the examples only as positive labels on numbers that happened to come up, not as draws from the concept, so every concept consistent with the examples stays equally plausible regardless of its size (no size principle). "Yes" is the prior-weighted share of the still-consistent concepts that contain the target, with an occasional example tolerated as an error; small, tight concepts get no advantage, so generalization is far broader than under strong sampling.

### example_noise_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 51.7 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle) with occasional guessing, but how readily they treat an example as a stray "noise" number that does not belong to the concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise rate: a high rate lets a rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42) and a low rate forces a concept that covers every example, so it changes which concept people infer from sets containing an outlier.

### inferred_noise_concept_learning — rejected (experiment1 round 1 candidate 4 refine incumbent concept_learning_with_guessing)

**Outcome:** predicts like existing model 'example_noise_concept_learning' (p_yes RMSE 0.00131 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of example_noise_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle), with the same guessing on a fraction of trials, but the share of examples they believe are random noise rather than drawn from the concept is not fixed at 10% — it is a property of their reasoning that we infer. The single change is this inferred noise belief: the more examples people treat as possibly random, the more readily a rule survives an example that violates it and the weaker the size principle's pull toward the tightest concept, changing generalization on sets with an odd-one-out example.

### inferred_outlier_rate_concept_learning — rejected (experiment1 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'inferred_example_noise_belief' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of inferred_example_noise_belief, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 with the size principle, but how often they believe an example is a stray, random number rather than a true member of the concept is not fixed at 10% — it is a property of the learner, inferred from the data. The one change is this inferred outlier rate: a high rate lets a concept that fits most (but not all) of the examples keep posterior weight and weakens how sharply the examples narrow generalization, whereas a low rate makes a single misfit example rule a concept out.

### tempered_evidence_concept_guessing — pruned (experiment1 end of experiment)

**Outcome:** 27.4 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with the same occasional guessing, but they do not weigh the examples as fully independent random draws from the concept — each example's evidence counts for an inferred fraction (or multiple) of a full draw. The one change is this tempered size principle: weighting the evidence below one keeps broad rules and wide intervals plausible and widens generalization, while weighting it above one makes people lock onto the tightest concept consistent with the examples.

### decision_gain_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 49.7 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not report their belief that the target is in the concept directly — they turn it into a yes/no decision with an inferred decision gain on the odds. The one change is this gain: above one, people maximize (beliefs above even odds become near-certain yeses, below even odds near-certain noes); below one, they hedge, so weak beliefs (targets just outside plausible concepts) get more yeses and strong beliefs (the examples themselves) fewer, symmetrically in log-odds rather than as a flat guessing floor.

### win_stay_lose_shift_concept_tracking — pruned (experiment1 end of experiment)

**Outcome:** 23.3 nats behind criterion_shifted_redundant_examples (2.7× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they hold a single running guess that they revise example by example, in the order shown (win-stay, lose-shift). After the first example they pick a concept in proportion to its Bayesian plausibility (weighted rules and intervals, size principle), keep it as long as each new example fits it, and only when an example contradicts it do they switch to a fresh concept drawn from the posterior given all examples seen so far — so broad concepts adopted early survive later examples that a full Bayesian learner would use to narrow them, and the same numbers in a different order generalize differently.

### digit_grid_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 445.2 nats behind criterion_shifted_redundant_examples (13.3× dse)

**Hypothesis:** People judge a target by its similarity to the examples in the written-digit form of the numbers: each number is a point on a grid of its tens digit and its ones digit, and a target's similarity to an example falls off exponentially with the attention-weighted city-block distance between them on that grid (with an inferred share of attention given to the ones versus the tens digit). The more similar the target is on average to the examples, the more likely a "yes" — so a target sharing the examples' final digit (e.g. 52 after 92, 82, 42) or decade looks similar even when it is far away numerically, without any rule being entertained.

### response_bias_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 55.0 nats behind criterion_shifted_redundant_examples (3.6× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest rule-and-interval model (42 weighted rules and all intervals of 1-100, size principle, 10% noisy examples), but they do not report their belief directly: they compare the evidence for "the program makes this number" against a shifted decision criterion, a response bias that adds a constant amount to the log-odds of every belief. A liberal criterion raises yeses most for middling and weak beliefs (targets just outside plausible concepts) while leaving near-certain beliefs nearly unchanged; unlike a guessing floor it does not lower yeses to the examples themselves, and unlike a decision gain it shifts all beliefs in one direction rather than pushing them apart or together.

### redundant_examples_tempered_concepts — pruned (experiment1 end of experiment)

**Outcome:** 25.0 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence and occasional guessing, but they treat additional examples as partly redundant rather than as independent draws, so the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it. The one change is this diminishing (or accelerating) return to set size: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen it much less than strong sampling predicts; an exponent of one recovers the incumbent.

### sublinear_evidence_accumulation — rejected (experiment1 round 2 candidate 4 refine incumbent tempered_evidence_concept_guessing)

**Outcome:** predicts like existing model 'redundant_examples_tempered_concepts' (p_yes RMSE 0.00130 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of redundant_examples_tempered_concepts, not a new hypothesis.

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with tempered evidence and occasional guessing, but the weight each example's evidence carries depends on how many examples they were shown — evidence accumulates with diminishing (or accelerating) returns rather than at a fixed rate per example. The one change is this inferred set-size dependence of the evidence weight: with diminishing returns, a single example already sharpens the concept strongly while a third or fourth example adds little, so people generalize more narrowly from one example and more broadly from four than a fixed per-example weight predicts; no dependence recovers the incumbent.

### tempered_evidence_decision_gain — pruned (experiment1 end of experiment)

**Outcome:** 13.6 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** Refinement of decision_gain_concept_learning: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples) and turn their belief into a yes/no answer through the same inferred decision gain on the log-odds, but they do not weigh the examples as fully independent draws from the concept — the one change, taken from the incumbent, is that each example's evidence counts for an inferred fraction (or multiple) of a full draw. Tempering thus decides which concepts people entertain (how broadly they generalize), while the decision gain alone decides how they hedge between belief and answer, with no flat guessing floor: targets with near-zero belief still get near-zero yeses, unlike under the incumbent's guessing.

### tempered_evidence_inferred_noise — rejected (experiment1 round 2 candidate 4 refine incumbent tempered_evidence_concept_guessing repair 1)

**Outcome:** too slow to fit: the fit of 'tempered_evidence_inferred_noise' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence (each example counts for an inferred fraction or multiple of a full draw) and occasional guessing, but how often they believe an example is a stray random number rather than a member of the concept is inferred from the data instead of fixed at 10%. The one change, taken from example_noise_concept_learning, is this inferred example-noise rate: tempering sets how sharply consistent concepts are favoured by their size, while the noise rate separately sets how readily a concept survives an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42), so sets with an outlier generalize differently from what either change alone predicts.

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 12.3 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted mathematical rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain turning belief into a yes/no answer), but their space of rules also contains offset arithmetic progressions — "every k-th number starting from r" (numbers leaving remainder r when divided by k, for steps 3 to 12), such as 1, 7, 13, 19, ... — and not only multiples counted from zero. These progressions share an inferred, a-priori small slice of prior belief, equally divided among them, so a set whose examples happen to fall on a common step (e.g. 85, 19, 91 all one more than a multiple of 6) generalizes to the other members of that progression.

### exemplar_averaged_generalization — pruned (experiment1 end of experiment)

**Outcome:** 285.1 nats behind criterion_shifted_redundant_examples (8.7× dse)

**Hypothesis:** People do not infer one concept from the whole set of examples; instead each example is stored as a separate exemplar, and the target is judged by how strongly each single stored example on its own generalizes to it — through the familiar number categories (mathematical rules and numeric ranges) that the example and target both belong to, with smaller categories counting more (the size principle applied to one example at a time). These single-example generalization strengths are averaged across the examples, GCM-style, and the average belief is turned into a yes/no answer, so adding consistent examples never sharpens generalization the way joint Bayesian inference does: a target that only some examples point to (e.g. a multiple of 4 after 16, 8, 2, 64) still gets substantial support.

### predictive_representativeness_judgment — pruned (experiment1 end of experiment)

**Outcome:** 96.9 nats behind criterion_shifted_redundant_examples (4.5× dse)

**Hypothesis:** People do not ask whether the target belongs to the concept they have inferred; they ask how representative the target would be as the program's next output — how much more likely the program is to produce exactly this number than a number picked at random, given the concepts the examples suggest (weighted rules and intervals, size principle). Their "yes" rises smoothly with this predicted probability of the target being generated next, so a target in a small, tight concept (32 after 16, 8, 64) is endorsed more readily than an equally certain member of a large concept (a number after examples suggesting "even" or a wide interval), unlike a membership judgment that treats all members of a believed concept alike.

### tempered_gain_shifted_criterion — rejected (experiment1 round 3 candidate 4 refine incumbent tempered_evidence_decision_gain)

**Outcome:** predicts like existing model 'criterion_shifted_tempered_decision' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of criterion_shifted_tempered_decision, not a new hypothesis.

**Hypothesis:** Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples) with tempered evidence, and turn their belief into a yes/no answer through the same inferred decision gain on its log-odds, but they compare that scaled evidence against a shifted decision criterion rather than against even odds. The one change, taken from response_bias_concept_learning, is this inferred criterion shift: a liberal (or conservative) criterion moves every belief's log-odds in one direction, so yeses to weakly supported targets rise (or fall) without the symmetric hedging of the gain forcing the examples' own yes rate down by the same amount.

### win_stay_lose_shift_decision_gain — pruned (experiment1 end of experiment)

**Outcome:** 9.6 nats behind criterion_shifted_redundant_examples; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 9

**Hypothesis:** Refinement of win_stay_lose_shift_concept_tracking: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it), but they do not answer by occasionally guessing at random. The one change, taken from the incumbent, replaces the flat guessing floor with an inferred decision gain on the log-odds of their belief that the tracked concept contains the target: people hedge (or sharpen) symmetrically between belief and answer, so targets with near-zero belief under the order-dependent guess still get near-zero yeses.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 362.9 nats behind criterion_shifted_redundant_examples (5.8× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 58.1 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### belief_carryover_assimilation — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'belief_carryover_assimilation' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 weighted rules and all intervals of 1-100, size principle with evidence tempered separately for rules and ranges, a decision gain and criterion shift turning belief into a yes/no answer), but they do not fully reset their belief between trials: the belief they act on is partly carried over from the previous trial, so their confidence that the current target belongs to the concept is pulled, by an inferred inertia weight, toward the confidence they had about the previous trial's target. This sequential assimilation makes consecutive answers in a session more alike than the stimuli alone warrant (e.g. a "no"-worthy target asked right after a near-certain "yes" gets more yeses); zero inertia recovers the best model.

### disjunctive_sampled_concepts — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'disjunctive_sampled_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do not consult a full, graded belief over all concepts; instead they bring to mind a small handful of candidate concepts (mathematical rules or numeric ranges), each recalled in proportion to how plausible it is given the examples (size principle, tempered evidence), and answer "yes" if the target fits any one of the concepts they recalled, occasionally answering at random instead. Because the check is disjunctive, a target supported by even a modestly plausible concept gets many yeses (the more concepts recalled, the more), while the yes rate saturates for well-supported targets — an asymmetric belief-to-answer mapping rather than a symmetric gain and criterion shift on the log-odds.

### landmark_split_tempering — rejected (experiment2 round 0 candidate 3 refine incumbent split_rule_range_tempering)

**Outcome:** predicts like existing model 'split_rule_range_tempering' (p_yes RMSE 0.00075 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of split_rule_range_tempering, not a new hypothesis.

**Hypothesis:** Refinement of split_rule_range_tempering: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), weigh the examples' evidence by separate inferred strengths for rules and for ranges, and answer through the same decision gain and shifted criterion. The one change, taken from round_landmark_interval_prior, is that the ranges people entertain are not equally natural a priori: ranges whose boundaries fall on round decade landmarks (e.g. 1-50, 20-29, 61-100, "the 80s") get extra prior belief by an inferred factor, so the broad range-based generalization that weak range evidence allows tends to stop at, or extend out to, the nearest decade boundary rather than fading evenly with distance from the examples.

### landmark_ranges_split_tempering — rejected (experiment2 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'split_rule_range_tempering' (p_yes RMSE 0.00075 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of split_rule_range_tempering, not a new hypothesis.

**Hypothesis:** Refinement of round_landmark_interval_prior: people do the same Bayesian concept learning over 42 weighted mathematical rules and all ranges of 1-100, with extra prior belief (by an inferred factor) for ranges whose boundaries fall on round decade landmarks, and answer through the same decision gain and shifted criterion. The one change, taken from split_rule_range_tempering, is that the examples' evidence (the size principle) is tempered by a separate inferred weight for rules and for ranges: people can lock onto a tight rule from a few examples while treating the same examples as only weak evidence about where a range stops, so range-based generalization stays broad and extends out to the nearest round decade boundary rather than hugging the examples' span (equal weights recover round_landmark_interval_prior).

### response_perseveration_criterion — rejected (experiment2 round 3 candidate 3 refine incumbent rule_size_prior_anchored_evidence)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian concept learning (42 rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger-weighted first example, a decision gain and shifted criterion), but their yes/no criterion is not fixed across the session — it is pulled toward the answer they just gave. The one change is response perseveration: after answering "yes" the criterion shifts toward "yes" on the next trial, and after "no" toward "no", by an inferred amount on the log-odds, so the trial-independent incumbent's under-production of yes-after-yes runs is closed and targets with middling belief are judged differently depending on the previous answer; a zero shift recovers the incumbent.

### response_perseveration_anchored — rejected (experiment2 round 1 candidate 4 refine incumbent first_example_anchored_concepts)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of first_example_anchored_concepts: people do the same first-example-anchored Bayesian concept learning over weighted rules and numeric ranges and answer through the same decision gain and shifted criterion, but their yes/no criterion is not fixed across trials — it carries over from their own previous answer. The one change is response perseveration: having just said "yes" makes the next "yes" more likely (and having just said "no" makes the next "no" more likely) by an inferred shift of the criterion on the log-odds, independent of the stimulus, which produces the sequential yes/no carryover the trial-independent incumbent fails to reproduce; zero carryover recovers the incumbent.

### anchored_range_bounded_rules_2 — rejected (experiment2 round 1 candidate 4 refine incumbent first_example_anchored_concepts repair 1)

**Outcome:** predicts like existing model 'anchored_range_bounded_rules' (p_yes RMSE 0.00064 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of anchored_range_bounded_rules, not a new hypothesis.

**Hypothesis:** Refinement of first_example_anchored_concepts: people do the same first-example-anchored Bayesian concept learning over weighted rules and numeric ranges (concepts containing the first example shown get extra belief; evidence tempered separately for rules and ranges) and answer through the same decision gain and shifted criterion. The one change, taken from range_bounded_rules_split_tempering, is that their rules can also be bounded to a range — besides "multiples of 10" they entertain "multiples of 10 from 70 to 100", any unbroken run of a rule's members, taking an inferred share of each rule's prior — so when the examples cluster, people generalize a rule only near the examples and give far fewer yeses to rule-consistent targets far outside the examples' span, while the first example still anchors which concepts are believed.

### graded_exemplar_category_match — rejected (experiment2 round 2 candidate 0 lens 10)

**Outcome:** too slow to fit: the fit of 'graded_exemplar_category_match' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People store each familiar number category (mathematical rules such as "powers of 2" or "multiples of 10", and numeric ranges) as the collection of its member numbers, and judge both the examples and the target by exemplar similarity to those stored members rather than by all-or-none membership: a number fits a category to the degree that it lies close on the number line to the category's nearest stored member, with the fit falling off exponentially at an inferred rate. Categories whose stored members closely match all the examples — diluted by how many members the category stores — are retrieved most strongly, and the target is endorsed in proportion to its similarity to the stored members of the retrieved categories, so near-misses (30 after 16, 8, 64; 51 just past a range of examples) receive graded partial yeses instead of none.

### pedagogical_example_sampling — rejected (experiment2 round 2 candidate 1 lens 0 repair 1)

**Outcome:** too slow to fit: the fit of 'pedagogical_example_sampling' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People assume the program's examples were chosen by a helpful teacher rather than drawn at random from the concept: they reason that a teacher would pick, for each example, a member of the concept that most clearly points a naive learner toward that concept. So when inferring the concept they ask not just "does this concept contain the examples?" but "would someone who had this concept in mind have chosen these particular examples to show it?" — an inferred strength of this pedagogical reasoning makes examples that are diagnostic of a concept (rather than shared by many concepts) count as stronger evidence for it, and with zero strength people reason from ordinary random sampling.

### weber_fuzzy_range_membership — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'weber_fuzzy_range_membership' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger-weighted first example, a decision gain and shifted criterion), but when they check whether the target falls inside a numeric range they believe in, they perceive the target's magnitude imprecisely, with noise that grows in proportion to the number's size (Weber's law). The one change is this scalar magnitude noise: the edges of believed ranges become fuzzy, so targets just outside a range get some yeses and targets just inside lose some, and this blurring is wider for large numbers (around 80-100) than for small ones (around 5-15), while membership in arithmetic rules is judged exactly; zero noise recovers the best model.

### response_perseveration_anchored_concepts — rejected (experiment2 round 3 candidate 4 refine incumbent rule_size_prior_anchored_evidence)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian concept learning (42 weighted rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger-weighted first example, a decision gain and shifted criterion), but their yes/no criterion is not fixed across the session — it is pulled toward the answer they gave on the immediately preceding trial. The one change is this response perseveration: after a "yes" the criterion becomes more liberal and after a "no" more conservative, by an inferred amount on the log-odds scale, so answers carry over from one trial to the next independently of the stimulus (zero stickiness recovers the refined model).

### stray_example_rate_anchored_concepts — rejected (experiment2 round 3 candidate 3 refine incumbent rule_size_prior_anchored_evidence repair 1)

**Outcome:** predicts like existing model 'rule_size_prior_anchored_evidence' (p_yes RMSE 0.00131 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_size_prior_anchored_evidence, not a new hypothesis.

**Hypothesis:** Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian concept learning (42 rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger-weighted first example, a decision gain and shifted criterion), but how often they believe the program emits a stray number that does not belong to its concept is their own inferred belief rather than a fixed 10%. The one change, taken from inferred_example_noise_belief, is this inferred stray-example rate: a high rate lets a tight rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42, or "powers of 2" after 16, 8, 2, 30) and flattens the size advantage of tiny concepts, while a low rate forces every example to be covered, so it changes which concept people infer mainly on sets containing an odd-one-out; a 10% rate recovers the incumbent.

### inferred_noise_rule_size_anchored — rejected (experiment2 round 3 candidate 4 refine incumbent rule_size_prior_anchored_evidence repair 1)

**Outcome:** predicts like existing model 'rule_size_prior_anchored_evidence' (p_yes RMSE 0.00120 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_size_prior_anchored_evidence, not a new hypothesis.

**Hypothesis:** Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian concept learning (42 weighted rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger- or weaker-weighted first example, a decision gain and shifted criterion), but how often they believe the program emits a stray number that does not belong to its concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise belief: a high believed noise rate lets a tight rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42, so 52 and 62 get yeses) and weakens how much a lone mismatching example rules a concept out, while a low rate forces a concept that covers every example (here a wide range or "contains a 2"); a rate of 10% recovers the refined model.
