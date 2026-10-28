# Tried before

39 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 268.5 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 289.6 nats behind scale_end_avoiding_ranges (10.7× dse)

**Hypothesis:** People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

### arithmetic_step_lattice — pruned (experiment1 end of experiment)

**Outcome:** 251.6 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

### tempered_evidence_interval — pruned (experiment1 end of experiment)

**Outcome:** 41.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

### erlang_interval_size_prior — pruned (experiment1 end of experiment)

**Outcome:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (size principle, 10% of examples random, "yes" = posterior probability the concept contains the target), but their prior over intervals depends on how wide an interval is rather than where it lies. Following Tenenbaum's original number game, interval width has an Erlang prior with an inferred scale, so people a priori favour moderately sized ranges and disfavour both tiny and near-total ranges, which controls how far they generalize beyond the examples' span.

### noise_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 43.9 nats behind scale_end_avoiding_ranges (5.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

### few_sample_majority_vote — pruned (experiment1 end of experiment)

**Outcome:** 32.9 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

### few_sample_majority_concept — rejected (experiment1 round 1 candidate 1 lens 4)

**Outcome:** predicts like existing model 'few_sample_majority_vote' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of few_sample_majority_vote, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and intervals of 1-100 (size principle, a width-based Erlang prior on intervals), but instead of reporting the posterior probability that the target belongs to the concept, they decide from a small number of hypotheses sampled from that posterior and go with the majority. This resource limit distorts the normative judgement by a single response-gain: posterior beliefs are pushed toward certain "yes"/"no" (few samples, majority rule) or compressed toward 50/50, with the degree of distortion inferred from the data.

### minimal_consistent_concept — pruned (experiment1 end of experiment)

**Outcome:** 240.9 nats behind scale_end_avoiding_ranges (10.8× dse)

**Hypothesis:** People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

### erlang_interval_response_lapse — pruned (experiment1 end of experiment)

**Outcome:** 31.3 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

### erlang_interval_with_response_lapse — rejected (experiment1 round 1 candidate 4 refine incumbent erlang_interval_size_prior)

**Outcome:** predicts like existing model 'erlang_interval_response_lapse' (p_yes RMSE 0.00062 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_response_lapse, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 (size principle, Erlang prior over interval width), but on some fraction of trials they do not consult their posterior at all and instead answer by a fixed yes/no guessing bias. The one change is this response lapse, which explains why people say "no" even to targets that are among the examples (about a third of the time here) and "yes" to targets no plausible concept contains, rather than forcing the concept posterior itself to absorb those answers.

### located_sized_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 35.8 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

### recency_weighted_evidence — pruned (experiment1 end of experiment)

**Outcome:** 33.4 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

### log_scale_erlang_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 53.4 nats behind scale_end_avoiding_ranges (5.4× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

### primacy_weighted_evidence — pruned (experiment1 end of experiment)

**Outcome:** 32.7 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

### span_scaled_hull_similarity — pruned (experiment1 end of experiment)

**Outcome:** 283.2 nats behind scale_end_avoiding_ranges (11.1× dse)

**Hypothesis:** People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

### noisy_criterion_threshold — pruned (experiment1 end of experiment)

**Outcome:** 38.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

### gamma_shape_interval_width_prior — pruned (experiment1 end of experiment)

**Outcome:** 32.5 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

### gamma_shape_interval_width_prior_2 — rejected (experiment1 round 2 candidate 4 refine incumbent erlang_interval_size_prior)

**Outcome:** predicts like existing model 'gamma_shape_interval_width_prior' (p_yes RMSE 0.00137 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of gamma_shape_interval_width_prior, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over how wide a range is takes a Gamma shape whose shape is inferred, instead of the Erlang's fixed shape of two. The one change is this free shape: if people's width prior falls off from the narrowest ranges (shape near or below one) rather than peaking at a moderate width, they believe the range ends close to the outermost examples, so they say "yes" to numbers inside the examples' span but sharply less to numbers just beyond it.

### tempered_evidence_erlang_width — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'erlang_interval_size_prior' (p_yes RMSE 0.00082 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_prior, not a new hypothesis.

**Hypothesis:** Refinement of `tempered_evidence_interval`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they weigh the examples' evidence at an inferred strength, raising the size-principle likelihood to a power rather than using it at exactly full strength. The one change is the interval prior: instead of a Beta density over where an interval lies, intervals get an Erlang prior over their width (from `erlang_interval_size_prior`), so evidence strength and an expected range width are separated. The claim is that people over-weight the examples relative to the ideal learner (a power above one), which makes them commit to ranges that just cover the examples, giving a sharper drop in "yes" at the edge of the examples' span than the ideal learner predicts, as the critique of the best model reports.

### endpoint_marking_interval_sampling — pruned (experiment1 end of experiment)

**Outcome:** 30.8 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but when two or more distinct examples are shown they assume the program's examples tend to mark the ends of its range (edge-marking, pedagogical sampling). The one change is an inferred evidence bonus for each end of a candidate range that coincides with the smallest or largest example, so the range that exactly spans the examples is favoured over wider ones, and people say "yes" inside the examples' span but sharply less just beyond it.

### decade_bounded_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `located_sized_interval_prior`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but their prior over where a range lies is not a smooth density over its midpoint: it favours ranges bounded by round decade numbers ("the 20s", "1 to 50", "51 to 100") by an inferred factor. The one change is replacing the Beta-over-midpoint location prior with this round-boundary preference, so generalization from examples tends to stop at decade boundaries — "yes" stays high up to the edge of the decade the examples fall in and drops sharply past it, a sharper edge near the examples' span than a width-only prior predicts.

### offset_arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 34.5 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning as in the best model (42 arithmetic rules and every interval of 1-100, with an Erlang width prior and a preference for decade-bounded ranges, size principle, "yes" = posterior probability the concept contains the target), but their space of candidate rules also includes offset arithmetic progressions — "every 4th number starting from 3" (3, 7, 11, ...), "every 6th number from 5" (5, 11, ..., 77, 83) — for every step from 3 to 12 and every non-zero starting offset. These progressions together receive an inferred share of prior belief, split equally among them, so examples whose gaps share a common step lead people to say "yes" to targets that continue the same counting pattern, even when they fall outside the examples' range and no multiple, digit or interval rule covers them.

### shepard_exemplar_similarity_average — pruned (experiment1 end of experiment)

**Outcome:** 289.5 nats behind scale_end_avoiding_ranges (10.3× dse)

**Hypothesis:** People do not infer one concept from the examples jointly; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars (an exemplar-memory account). The similarity of a target to one exemplar is Shepard's consequential-region generalization: how likely a familiar number concept (an arithmetic rule, or a range with a preference for moderate widths) that contains that single exemplar also contains the target. Because each exemplar generalizes on its own and the similarities are averaged rather than combined as joint evidence, adding examples that share a narrow rule (e.g. 4, 16, 12) does not sharpen generalization the way it would for an ideal learner.

### weber_noisy_range_membership — pruned (experiment1 end of experiment)

**Outcome:** 31.8 nats behind scale_end_avoiding_ranges (4.2× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (size principle, a prior favouring moderately wide ranges), but they check whether a target falls inside a range on a noisy analog mental number line whose noise grows in proportion to magnitude (Weber's law), while rule membership (even, multiple of 5, ...) is checked exactly. So the edge of a range is blurred rather than sharp: targets just outside the range sometimes count as "in" and targets just inside sometimes as "out", and this blur is wider among large numbers than among small ones.

### per_endpoint_round_boundary_prior — pruned (experiment1 end of experiment)

**Outcome:** 33.5 nats behind scale_end_avoiding_ranges (4.4× dse)

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) and they favour ranges with round boundaries, but they judge roundness one endpoint at a time and under either everyday convention. The one change: each end of a range that falls on a round decade number — a lower end like 1, 10, 11, 20, 21, ... or an upper end like 9, 10, 19, 20, ... 100 (so "the 20s" as 20-29, 21-30 or 20-30) — multiplies the range's prior by an inferred factor, so a range with one round end and one end hugging the examples is also favoured, instead of only ranges whose both ends sit at the incumbent's strict 21-30 convention.

### per_end_decade_boundary_prior — rejected (experiment1 round 3 candidate 4 refine incumbent decade_bounded_interval_prior)

**Outcome:** predicts like existing model 'erlang_interval_size_prior' (p_yes RMSE 0.00085 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_prior, not a new hypothesis.

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) with a prior preference for ranges bounded by round decade numbers, but each end of a range is judged round or not on its own: a range gains the inferred round-boundary factor once for a round lower end (1, 11, 21, ...) and once more for a round upper end (10, 20, ..., 100), instead of only when both ends are round. The one change is this per-end preference, so people can stop generalizing at the decade edge on one side of the examples (e.g. "up to 50") while extending past it on the other, rather than favouring only whole-decade blocks.

### anchored_endpoint_or_decade_ranges — pruned (experiment1 end of experiment)

**Outcome:** 30.6 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `endpoint_marking_interval_sampling`: people do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they assume the program's examples tend to mark the ends of its range — but a range's ends can also be anchored at salient round numbers. The one change is grafting the incumbent's round-decade boundary preference onto the endpoint-marking model: ranges bounded by decade numbers ("the 20s", "1 to 50") get an inferred prior bonus alongside the bonus for ends coinciding with the smallest or largest example, so generalization stops either exactly at the examples' span or at the edge of the enclosing decade, whichever anchors are available.

### scale_edge_range_prior — rejected (experiment1 round 4 candidate 3 refine incumbent family_pooled_rule_prior)

**Outcome:** predicts like existing model 'scale_end_avoiding_ranges' (p_yes RMSE 0.00060 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of scale_end_avoiding_ranges, not a new hypothesis.

**Hypothesis:** Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior share for each kind of rule split equally within the kind, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target). The one change: people do not treat the ends of the 1-100 scale as neutral range boundaries — a range that runs all the way to 1 or to 100 ("numbers up to 20", "numbers above 70") has its prior multiplied by an inferred factor, so when the examples lie near an end of the scale, how far people generalize toward that end depends on whether they avoid (or favour) ranges that hit the scale's edge, not just on the range's width.

### size_graded_family_rule_prior — rejected (experiment1 round 4 candidate 4 refine incumbent family_pooled_rule_prior)

**Outcome:** predicts like existing model 'family_pooled_rule_prior' (p_yes RMSE 0.00167 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of family_pooled_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior share for each kind of rule, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but within a kind they do not treat all rules as equally likely: a rule's prior grows (or shrinks) with how many numbers it contains, by an inferred power. The one change is this size-graded within-kind prior — e.g. "multiples of 3" or "even numbers", which people meet constantly, are a priori more plausible than "multiples of 11" or "cubes" — so it partly offsets the size principle and makes people generalize to broader rules of the same kind more readily than the equal-split prior predicts.

### parity_kind_rule_prior_2 — rejected (experiment1 round 4 candidate 4 refine incumbent family_pooled_rule_prior repair 1)

**Outcome:** predicts like existing model 'parity_kind_rule_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of parity_kind_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, a prior share for each kind of rule split equally among that kind's rules, "yes" = posterior probability the concept contains the target). The one change is that people treat parity — "even numbers" and "odd numbers" — as a kind of rule of its own, with its own prior share, rather than lumping it with the special number sets (primes, squares, cubes). So "even numbers" no longer has to be exactly as plausible a priori as "primes" or "cubes", and how readily people jump from a few even examples to "all even numbers" is set separately from how readily they jump to squares or primes.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 42.5 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Outcome:** 39.3 nats behind scale_end_avoiding_ranges (4.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### serial_position_example_forgetting — rejected (experiment2 round 0 candidate 2 lens 6)

**Outcome:** predicts like existing model 'random_example_forgetting' (p_yes RMSE 0.00142 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of random_example_forgetting, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (kind-level rule prior, size principle, a width prior with a preference for decade-bounded ranges), but they generalize only from the examples still in working memory, and memory follows the order of presentation: the last example shown is always held, and each earlier example is lost with a probability that grows the further back it was shown (retention decays geometrically with the number of examples that came after it). So the same examples shown in a different order lead to different generalization, pulled toward the concepts the most recent examples suggest.

### rule_region_width_prior — rejected (experiment2 round 0 candidate 3 refine incumbent range_restricted_rule_concepts)

**Outcome:** predicts like existing model 'range_restricted_rule_concepts' (p_yes RMSE 0.00106 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of range_restricted_rule_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, ranges of 1-100 and range-restricted rules ("even numbers in the 60s-70s"), with the size principle and "yes" = posterior probability the concept contains the target, but the region that restricts a rule has its own width prior, separate from the prior over plain ranges. The one change is a second inferred typical width for rule-restricting regions, so how far people extend an arithmetic pattern beyond the examples' span (e.g. "yes" to 30 or 90 after 66,78) is governed independently of how far they extend a plain numeric range.

### region_specific_restricted_rule_width — rejected (experiment2 round 0 candidate 4 refine incumbent range_restricted_rule_concepts)

**Outcome:** predicts like existing model 'range_restricted_rule_concepts' (p_yes RMSE 0.00175 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of range_restricted_rule_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, plain ranges of 1-100 and range-restricted rules ("even numbers in the 60s-70s", "multiples of 4 below 20"), with the size principle and "yes" = posterior probability that the concept contains the target. The one change is that the region a rule is restricted to has its own size prior: people expect how wide a rule's region is to be set separately from how wide a plain range is, with its own inferred typical width (and decade-boundary preference), instead of reusing the plain-range prior. So how sharply "yes" to rule members drops off outside the examples' region, such as 12 or 30 after 66,78, is no longer tied to how far plain-range generalization extends.

### graded_upper_edge_marking_ranges — rejected (experiment2 round 2 candidate 3 refine incumbent upper_edge_marking_ranges)

**Outcome:** predicts like existing model 'upper_edge_marking_ranges' (p_yes RMSE 0.00152 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upper_edge_marking_ranges, not a new hypothesis.

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and they take the largest example as marking where the program's range stops at the top — but only approximately. The one change: the inferred evidence bonus is not reserved for a range whose top is exactly the largest example; a range ending one, two or three numbers above it gets a share of the bonus that shrinks by an inferred factor with each step of the gap, so people expect the range to stop at or just above the largest example, accepting the next number or two up more readily than numbers further up, rather than placing a sharp edge exactly at the largest example.

### top_edge_marking_all_ranges — rejected (experiment2 round 2 candidate 4 refine incumbent upper_edge_marking_ranges)

**Outcome:** predicts like existing model 'upper_edge_marking_ranges' (p_yes RMSE 0.00056 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upper_edge_marking_ranges, not a new hypothesis.

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules ("even numbers from 60 to 78"), and they take the largest example as marking where the program's range stops at the top. The one change: this edge inference applies to every range people consider, including the range that restricts a rule, not only to plain ranges, so a range-restricted rule whose upper end is exactly the largest example gets the same inferred evidence bonus. The model therefore predicts that people reject rule members above the largest example (after 66,78 they say no to 80 or 90 more than to 60 or 50) as well as plain neighbours above it.

### upper_edge_discounted_range_size_2 — rejected (experiment2 round 2 candidate 4 refine incumbent upper_edge_marking_ranges repair 1)

**Outcome:** predicts like existing model 'upper_edge_discounted_range_size' (p_yes RMSE 0.00027 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upper_edge_discounted_range_size, not a new hypothesis.

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and they take the largest example as marking where the program's range stops at the top, but they apply the size principle only partly to numeric ranges. The one change, grafted from `range_discounted_size_principle`: when judging how well a range explains the examples, people treat it as wider than it is (its effective size lies between its true width and the whole 1-100 scale, by an inferred degree), so with more examples "yes" spreads further below and across the examples' neighbourhood than strong sampling predicts, while generalization above the largest example stays cut off by the upper-edge inference.
