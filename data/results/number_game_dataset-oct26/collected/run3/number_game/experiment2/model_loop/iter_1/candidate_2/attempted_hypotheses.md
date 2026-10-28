# Tried before

39 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_feature_similarity_gcm — pruned (experiment1 end of experiment)

**Outcome:** 77.4 nats behind session_growing_acquiescence_concepts (3.9× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

### shared_property_overlap — pruned (experiment1 end of experiment)

**Outcome:** 68.7 nats behind session_growing_acquiescence_concepts (4.1× dse)

**Hypothesis:** People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

### serial_position_weighted_concepts — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00118 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over arithmetic rules and numeric intervals (size principle), but they do not weigh the examples equally: how strongly each example counts as evidence depends on its serial position in the order shown, with later (recency) or earlier (primacy) examples dominating the inferred concept. The total evidence is fixed by the number of examples and only its distribution across positions changes, so a single example gives ordinary concept learning, while reordering the same examples changes which targets get a "yes".

### fatigue_lapse_interval_concepts — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_interval)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people answer by the same Bayesian concept learning over 42 arithmetic rules and all intervals of 1-100 (rule weights inferred, interval prior a Beta density over the midpoint, size principle), but on some trials they stop consulting the concept and answer yes or no at random, and these random lapses become more frequent as the session wears on (fatigue). The single change is this time-dependent lapse, which both softens the model's near-certain predictions and produces the observed drift toward more yeses late in the session.

### bayesian_concepts_drifting_guess — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_interval)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people judge by the same Bayesian concept learning over rules and midpoint-weighted intervals, but on some trials they do not consult the concept posterior and instead guess, and the yes-rate of that guessing drifts over the session as time on task grows. The one change is this session-drifting guess process mixed into the response, which addresses the critique that yes responses become more frequent later in the session.

### bp_rules_erlang_interval_size — pruned (experiment1 end of experiment)

**Outcome:** 186.9 nats behind session_growing_acquiescence_concepts (6.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

### periodic_spacing_concepts — rejected (experiment1 round 0 candidate 2 lens 2 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.143 > 1.05; min bulk ESS 19 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.99), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People learn the hidden program's concept by Bayesian inference with the size principle, but besides named arithmetic rules and numeric ranges they also consider any regular spacing the examples suggest: "every d-th number starting from r" for any step d (e.g. 3, 7 suggests 3, 7, 11, 15, ...; 20, 32, 92 suggests every 12th number from 8), with shorter steps more plausible a priori. Targets that continue the examples' common spacing therefore get "yes" even when no named rule (multiple of k, ending digit) covers them.

### conservative_evidence_interval_concepts — pruned (experiment1 end of experiment)

**Outcome:** 30.9 nats behind session_growing_acquiescence_concepts (3.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

### size_principle_strength_intervals — pruned (experiment1 end of experiment)

**Outcome:** 57.8 nats behind session_growing_acquiescence_concepts (5.0× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

### first_example_weak_sampling_concepts — pruned (experiment1 end of experiment)

**Outcome:** 28.8 nats behind session_growing_acquiescence_concepts (2.8× dse)

**Hypothesis:** People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

### overestimated_randomness_concepts — dropped (experiment1 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'overestimated_randomness_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People learn the program's concept by ideal Bayesian inference over arithmetic rules and numeric intervals with the size principle, but they believe the program is far less reliable than it is: they think a sizeable, inferred share of its outputs are arbitrary numbers unrelated to its rule. Because any single example might be such a random output, it rules out the concepts that do not contain it only weakly, so people generalize broadly — saying "yes" to many unrelated targets — after one example, and tighten only as consistent examples accumulate.

### range_scaled_magnitude_gradient — pruned (experiment1 end of experiment)

**Outcome:** 135.1 nats behind session_growing_acquiescence_concepts (6.0× dse)

**Hypothesis:** People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

### sample_size_growing_evidence_concepts — pruned (experiment1 end of experiment)

**Outcome:** 15.8 nats behind session_growing_acquiescence_concepts (3.1× dse)

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

### iter1_candidate4 — pruned (experiment1 end of experiment)

**Outcome:** 23.1 nats behind session_growing_acquiescence_concepts (3.0× dse)

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

### sampling_assumption_inference_intervals — dropped (experiment1 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'sampling_assumption_inference_intervals' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `size_principle_strength_intervals`: people do the same Bayesian concept learning over the 42 arithmetic rules and the midpoint-weighted intervals of 1-100, but instead of applying a fixed partial size principle to every example, they are uncertain whether the examples were deliberately sampled from the concept (strong sampling) or merely happen to be consistent with it (weak sampling), and they infer this from the examples themselves. The one change replaces the per-example size-principle strength with this set-level sampling-assumption inference (as in Navarro et al., 2012): a single example is weak evidence that sampling is strong, so people generalize broadly from one example, while each further example that tightly fits a small concept makes strong sampling — and hence narrow, size-principle generalization — more credible.

### recency_forgetting_concepts — rejected (experiment1 round 2 candidate 0 lens 6)

**Outcome:** too slow to fit: the fit of 'recency_forgetting_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but they reason only from the examples they still hold in memory when the target is asked: the last example shown is always remembered, while earlier examples drop out of memory with a probability that grows the further back in the sequence they were shown. A forgotten example neither rules out concepts nor sharpens them, so generalization is driven mostly by the most recent examples, and reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than ones fitting only the last).

### digit_place_similarity_gradient — pruned (experiment1 end of experiment)

**Outcome:** 59.8 nats behind session_growing_acquiescence_concepts (3.0× dse)

**Hypothesis:** People judge a target by its similarity to the examples in their written decimal form rather than on the number line: they compare the target with each example place by place, and every mismatch in the units digit or in the tens digit multiplies similarity down by its own inferred factor. The more similar the target is on average to the examples under this digit-place metric, the more likely people are to say "yes" — so 21 after 91, or 58 after 28 and 98, is endorsed despite being numerically far away, while 88 after 89 is rejected despite being adjacent.

### soft_criterion_concept_decision — pruned (experiment1 end of experiment)

**Outcome:** 22.5 nats behind session_growing_acquiescence_concepts (2.0× dse)

**Hypothesis:** People infer the hidden concept by plain Bayesian concept learning over arithmetic rules and numeric intervals (the simplest current inference, with the size principle), but they do not report that belief by probability matching: they answer "yes" when their confidence that the target belongs clears a personal criterion, applied noisily. How high the criterion sits (a bias toward "yes" or "no") and how sharply confidence is turned into a choice are the only changes, so moderately believed targets are pushed toward one answer and near-certain beliefs are softened toward the other.

### session_drifting_evidence_concepts — rejected (experiment1 round 2 candidate 3 refine incumbent sample_size_growing_evidence_concepts)

**Outcome:** too slow to fit: the fit of 'session_drifting_evidence_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with the number of examples, but how much weight they give the examples' evidence drifts over the session — as the task wears on, people scrutinise the examples less and so hold a flatter, broader posterior over concepts. The one change is that the evidence strength is scaled by an exponential drift in the trial's position in the session (rate inferred), which predicts more liberal, broader generalization (more "yes") late in the session, the discrepancy the incumbent's critique flags.

### session_drift_liberal_concepts — rejected (experiment1 round 2 candidate 4 refine incumbent sample_size_growing_evidence_concepts)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but as the session wears on they grow more liberal, increasingly accepting targets their concept does not contain. The one change is a "yes"-drift: with a probability that changes smoothly (logistically, slope inferred) with how far into the session the trial is, people answer "yes" regardless of the concept, which predicts the rising yes-rate across the session that the incumbent's critique flags (and reduces to a constant yes-bias if the slope is zero).

### first_example_weak_erlang_concepts — pruned (experiment1 end of experiment)

**Outcome:** 26.0 nats behind session_growing_acquiescence_concepts (2.7× dse)

**Hypothesis:** Refinement of `first_example_weak_sampling_concepts`: people update their concept one example at a time, and the first example only rules out concepts that do not contain it, while every later example is treated as drawn from the concept (size principle, weighted by an inferred strength) — but people also hold a prior preference for number ranges of intermediate size (Tenenbaum's Erlang size prior on intervals, scale inferred, grafted from `iter1_candidate4`). The one change is this size prior: because a single example carries no size-principle evidence under this account, what people generalize to from one example is set entirely by their prior, and the size prior predicts a moderately wide range around a lone example rather than the broad ranges near 1-100 that a size-blind prior favours.

### recency_decayed_evidence_concepts — pruned (experiment1 end of experiment)

**Outcome:** 35.9 nats behind session_growing_acquiescence_concepts (4.0× dse)

**Hypothesis:** People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but their memory of the examples fades with how long ago each was shown: the last example counts as full evidence, and each example further back in the sequence counts as progressively weaker evidence, by a constant inferred retention factor per position. Generalization is therefore driven mostly by the most recent examples, so reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than targets that fit only the last).

### session_drifting_yes_bias_concepts — rejected (experiment1 round 2 candidate 3 refine incumbent sample_size_growing_evidence_concepts repair 1)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with the number of examples, but their willingness to say "yes" given that belief drifts over the session — as the task wears on, people become more liberal and endorse targets they are only partly confident about. The one change is a response bias on the log-odds of the posterior belief that shifts linearly with the trial's position in the session (rate inferred, zero at mid-session), which predicts a higher yes-rate late in the session than early, the discrepancy the incumbent's critique flags.

### lapsing_growing_evidence_concepts — pruned (experiment1 end of experiment)

**Outcome:** 11.0 nats behind session_growing_acquiescence_concepts (2.5× dse)

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but on some trials they answer without consulting their concept at all, guessing "yes" or "no" with their own inferred rates. The one change is this attention-lapse response floor and ceiling (an inferred chance of a blind "yes" and a separate chance of a blind "no"), which predicts a steady baseline of "yes" to targets the concept clearly excludes and occasional "no" to targets it clearly contains, where the incumbent puts both at near-certainty.

### coherence_attention_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 101.5 nats behind session_growing_acquiescence_concepts (4.8× dse)

**Hypothesis:** People do not infer a concept; they judge the target by its similarity to the stored examples (exemplar generalization), but their attention is drawn to the dimensions on which the stored examples cohere: every property all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) gets attention in proportion to how distinctive it is in 1-100, so a target that lacks a rare shared property (e.g. "ends in 8", "in the 20s") is judged dissimilar to every example, while closeness on the number line counts on a scale that stretches with the examples' spread. Saying "yes" is a Luce choice between the target's mean similarity to the examples and a fixed criterion.

### transformational_distance_similarity — pruned (experiment1 end of experiment)

**Outcome:** 134.7 nats behind session_growing_acquiescence_concepts (6.0× dse)

**Hypothesis:** People do not induce a concept or measure closeness on the number line; they judge whether the target can be made from one of the examples by a short chain of simple mental operations on numbers (add or subtract one, add or subtract ten, double, halve, reverse the digits), so the fewer operations it takes to turn the examples into the target, the more likely they say "yes" (transformational similarity). This predicts "yes" to 32 after 16, to 61 after 16, and to 40 after 20 even though they are far away on the number line, while a numerically moderate jump that needs many steps (e.g. 47 after 20) is rejected.

### decade_aligned_interval_concepts — pruned (experiment1 end of experiment)

**Outcome:** 10.2 nats behind session_growing_acquiescence_concepts (3.3× dse)

**Hypothesis:** Refinement of `lapsing_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size and attention lapses, but they mentally carve the number line into decades, so number ranges whose edges fall on decade boundaries (e.g. 20-29, 30-59, 1-10) are a priori more plausible than arbitrary ranges by an inferred factor. The one change is this decade-aligned prior bonus on intervals, which predicts that a nearby target outside the examples' span is endorsed more often when it shares a decade with an example (25 after 21, 23) than when it lies just across a decade boundary (31 after 27, 29) — the same-decade bonus the incumbent's critique flags.

### win_stay_lose_sample_concepts — pruned (experiment1 end of experiment)

**Outcome:** 34.7 nats behind session_growing_acquiescence_concepts (3.9× dse)

**Hypothesis:** People do not keep a full belief over every concept; they read the examples one at a time while holding a single working guess (win-stay, lose-sample). After the first example they pick a concept in proportion to how well it explains that example; each later example that the guess contains leaves it in place, and only an example it fails to contain makes them drop it and draw a new guess from what all the examples seen so far support. They answer "yes" if their current guess contains the target (apart from occasional blind guesses), so consistent later examples do not narrow the guess the way an ideal learner's belief narrows, and the order of the examples changes which concept people end up holding.

### session_loosening_decade_range_concepts — rejected (experiment1 round 4 candidate 3 refine incumbent decade_aligned_range_concepts)

**Outcome:** too slow to fit: the fit of 'session_loosening_decade_range_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100, with evidence weight growing with set size, but as the session wears on they grow less strict, giving the examples' evidence progressively less weight so that their generalization broadens. The one change is that the evidence weight drifts with how far into the session the trial falls (an inferred drift rate), which predicts more "yes" late in the session than early — chiefly to targets near the examples or sharing a broad rule with them that a tight concept excludes — the session-liberalization discrepancy the incumbent's critique flags.

### decade_anchored_edge_ranges — rejected (experiment1 round 4 candidate 4 refine incumbent decade_aligned_range_concepts)

**Outcome:** predicts like existing model 'decade_aligned_range_concepts' (p_yes RMSE 0.00096 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of decade_aligned_range_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, and they think about number ranges in round decades — but they anchor each end of a range at a decade boundary independently, so a range earns the same inferred prior bonus once for a lower edge on a decade start and again for an upper edge on a decade end, instead of a bonus only when both ends are decade-aligned. The one change is this per-edge decade anchoring, which predicts that generalization stretches to the decade boundary on one side even when the range's other end is not decade-aligned (e.g. after 23, 25, 34 people endorse 38-39 more than the incumbent predicts, while 21-22 gets no matching boost).

### criterion_decision_decade_concepts — pruned (experiment1 end of experiment)

**Outcome:** 6.6 nats behind session_growing_acquiescence_concepts (2.0× dse)

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100, with evidence weight growing with set size, but they do not report their belief by probability matching — they say "yes" when their confidence that the target belongs clears a personal criterion, applied noisily (the decision rule grafted from `soft_criterion_concept_decision`). The one change is this criterion-based decision stage (an inferred criterion and decision sharpness), which predicts that targets the learner believes moderately are pushed toward a consistent answer, and that near-certain inclusions and exclusions are softened, rather than "yes" rates equal to the posterior probability everywhere.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 299.4 nats behind session_growing_acquiescence_concepts (6.5× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 212.5 nats behind session_growing_acquiescence_concepts (6.5× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Outcome:** 206.5 nats behind session_growing_acquiescence_concepts (6.9× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### overestimated_program_noise_concepts — rejected (experiment2 round 0 candidate 0 lens 4)

**Outcome:** predicts like existing model 'session_growing_acquiescence_concepts' (p_yes RMSE 0.00173 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of session_growing_acquiescence_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `session_growing_acquiescence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100 (evidence weight growing with set size, blind "yes" growing over the session), but they believe the hidden program is much less reliable than it is — that a sizeable, inferred share of its outputs are arbitrary numbers unrelated to its rule. The one change is this inferred believed-noise rate (fixed at 10% in the incumbent), which makes any single example only weak grounds to exclude concepts that do not contain it, predicting broader generalization (more "yes" to non-example targets) after one example, while several consistent examples still converge on a tight concept.

### set_size_gated_range_concepts — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'set_size_gated_range_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `session_growing_acquiescence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring number ranges (evidence weight growing with set size, a blind-"yes" rate that grows over the session), but how plausible they find a magnitude-range explanation at all depends on how many examples they see: a lone example has no spread to suggest a range, so with one example people reason almost entirely about what kind of number it is (odd/even, multiple, power, final digit), and range concepts gain prior credibility only as more examples arrive. The one change is this set-size-dependent prior share of rules versus ranges (an inferred baseline and slope), which predicts that after a single example people endorse far-away targets of the same kind (e.g. same parity) much more, and nearby targets of a different kind much less, than the incumbent predicts.

### first_example_partial_size_principle — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'first_example_partial_size_principle' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People revise their concept one example at a time, and the first example is read mainly as "the program can produce this number": it rules out concepts that do not contain it but only partly favours small concepts, because a lone number is no suspicious coincidence yet; every later example is treated as a deliberate sample, so the full size principle favours the tightest concepts that keep fitting (otherwise the same conservative Bayesian learner over arithmetic rules and decade-favoured ranges, with session-growing blind "yes"). Because which example comes first gets the weakened size principle, the order of examples matters, and a single example is generalized broadly to the large rules and ranges that contain it (e.g. 5 to other numbers ending in 5) while still sharply rejecting numbers that share nothing with it.

### anything_goes_hypothesis_concepts — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'anything_goes_hypothesis_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `session_growing_acquiescence_concepts`: people do the same conservative Bayesian concept learning (42 arithmetic rules plus midpoint-weighted, decade-favoured intervals, evidence weight growing with set size, a blind-"yes" rate that rises over the session), but they also entertain the possibility that the program simply produces any number at all — an "anything goes" random-number hypothesis held with its own inferred prior share, far larger than the share the full range 1-100 gets as one interval among thousands. The one change is this privileged unconstrained hypothesis: because it pays only a mild size-principle penalty for one example but is quickly ruled out as consistent examples accumulate, it predicts broad, distance-insensitive "yes" to far targets after a single example (the single-example generalization the critique flags) that vanishes for larger sets.

### single_example_rule_preference_concepts — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'single_example_rule_preference_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `session_growing_acquiescence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100 (evidence weight growing with set size, a blind-"yes" rate that grows over the session), but a lone example does not suggest a number range to them — with nothing to span, they look for an arithmetic property of that number instead, so when the set has a single example the rules receive an inferred extra share of prior belief at the expense of ranges. The one change is this single-example shift of prior belief from ranges toward rules, which predicts broader generalization from one example to far-away numbers sharing its property (e.g. 29 after 89, 35 after 5) while its immediate neighbours (88 after 89) stay rarely endorsed — the single-example non-member yes-rate the incumbent's critique flags as too low.
