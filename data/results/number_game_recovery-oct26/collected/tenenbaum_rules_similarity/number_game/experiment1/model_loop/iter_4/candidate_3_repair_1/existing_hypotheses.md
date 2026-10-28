# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## family_pooled_rule_prior  — rank 0, the best model on this data, ELPD-LOO -1247.2

Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (rule share, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but their prior over rules is set by the kind of rule, not rule by rule. The one change: people hold a belief about how likely each kind of rule is (powers, multiples, final digit, contained digit, and special sets such as primes, evens, odds, squares, cubes) and treat all rules of one kind as equally likely, so "multiples of 7" is as plausible a priori as "multiples of 4", instead of the incumbent's separate free weight for each of the 42 rules.

## decade_bounded_interval_prior  — rank 1, 29.4 ± 7.5 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1276.6

Refinement of `located_sized_interval_prior`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but their prior over where a range lies is not a smooth density over its midpoint: it favours ranges bounded by round decade numbers ("the 20s", "1 to 50", "51 to 100") by an inferred factor. The one change is replacing the Beta-over-midpoint location prior with this round-boundary preference, so generalization from examples tends to stop at decade boundaries — "yes" stays high up to the edge of the decade the examples fall in and drops sharply past it, a sharper edge near the examples' span than a width-only prior predicts.

## erlang_interval_size_prior  — rank 2, 29.4 ± 7.5 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1276.6

Refinement of bigelow_piantadosi_interval: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (size principle, 10% of examples random, "yes" = posterior probability the concept contains the target), but their prior over intervals depends on how wide an interval is rather than where it lies. Following Tenenbaum's original number game, interval width has an Erlang prior with an inferred scale, so people a priori favour moderately sized ranges and disfavour both tiny and near-total ranges, which controls how far they generalize beyond the examples' span.

## anchored_endpoint_or_decade_ranges  — rank 3, 29.5 ± 7.4 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1276.7

Refinement of `endpoint_marking_interval_sampling`: people do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they assume the program's examples tend to mark the ends of its range — but a range's ends can also be anchored at salient round numbers. The one change is grafting the incumbent's round-decade boundary preference onto the endpoint-marking model: ranges bounded by decade numbers ("the 20s", "1 to 50") get an inferred prior bonus alongside the bonus for ends coinciding with the smallest or largest example, so generalization stops either exactly at the examples' span or at the edge of the enclosing decade, whichever anchors are available.

## endpoint_marking_interval_sampling  — rank 4, 29.7 ± 7.5 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1277.0

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but when two or more distinct examples are shown they assume the program's examples tend to mark the ends of its range (edge-marking, pedagogical sampling). The one change is an inferred evidence bonus for each end of a candidate range that coincides with the smallest or largest example, so the range that exactly spans the examples is favoured over wider ones, and people say "yes" inside the examples' span but sharply less just beyond it.

## erlang_interval_response_lapse  — rank 5, 30.2 ± 7.5 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1277.5

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

## weber_noisy_range_membership  — rank 6, 30.7 ± 7.7 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1278.0

People do Bayesian concept learning over arithmetic rules and numeric ranges (size principle, a prior favouring moderately wide ranges), but they check whether a target falls inside a range on a noisy analog mental number line whose noise grows in proportion to magnitude (Weber's law), while rule membership (even, multiple of 5, ...) is checked exactly. So the edge of a range is blurred rather than sharp: targets just outside the range sometimes count as "in" and targets just inside sometimes as "out", and this blur is wider among large numbers than among small ones.

## gamma_shape_interval_width_prior  — rank 7, 31.4 ± 7.7 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1278.7

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

## primacy_weighted_evidence  — rank 8, 31.6 ± 7.5 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1278.9

People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

## few_sample_majority_vote  — rank 9, 31.8 ± 7.8 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1279.1

People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

## recency_weighted_evidence  — rank 10, 32.3 ± 7.9 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1279.6

People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

## per_endpoint_round_boundary_prior  — rank 11, 32.4 ± 7.7 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1279.6

Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) and they favour ranges with round boundaries, but they judge roundness one endpoint at a time and under either everyday convention. The one change: each end of a range that falls on a round decade number — a lower end like 1, 10, 11, 20, 21, ... or an upper end like 9, 10, 19, 20, ... 100 (so "the 20s" as 20-29, 21-30 or 20-30) — multiplies the range's prior by an inferred factor, so a range with one round end and one end hugging the examples is also favoured, instead of only ranges whose both ends sit at the incumbent's strict 21-30 convention.

## offset_arithmetic_progression_concepts  — rank 12, 33.4 ± 7.7 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1280.7

People do Bayesian concept learning as in the best model (42 arithmetic rules and every interval of 1-100, with an Erlang width prior and a preference for decade-bounded ranges, size principle, "yes" = posterior probability the concept contains the target), but their space of candidate rules also includes offset arithmetic progressions — "every 4th number starting from 3" (3, 7, 11, ...), "every 6th number from 5" (5, 11, ..., 77, 83) — for every step from 3 to 12 and every non-zero starting offset. These progressions together receive an inferred share of prior belief, split equally among them, so examples whose gaps share a common step lead people to say "yes" to targets that continue the same counting pattern, even when they fall outside the examples' range and no multiple, digit or interval rule covers them.

## located_sized_interval_prior  — rank 13, 34.7 ± 8.3 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1281.9

Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

## noisy_criterion_threshold  — rank 14, 37.5 ± 7.9 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1284.7

People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

## bigelow_piantadosi_interval  — rank 15, 38.2 ± 8.7 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.4

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## tempered_evidence_interval  — rank 16, 40.6 ± 8.9 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1287.8

A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

## bigelow_piantadosi_independent  — rank 17, 41.5 ± 8.6 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1288.7

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## noise_tolerant_concept_learning  — rank 18, 42.8 ± 8.7 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1290.0

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

## log_scale_erlang_interval_prior  — rank 19, 52.3 ± 9.9 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1299.6

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

## minimal_consistent_concept  — rank 20, 239.8 ± 22.3 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1487.0

People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

## arithmetic_step_lattice  — rank 21, 250.5 ± 23.7 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1497.8

People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

## featural_exemplar_similarity  — rank 22, 267.4 ± 25.4 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1514.7

People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

## span_scaled_hull_similarity  — rank 23, 282.1 ± 25.5 nats behind the best (11.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1529.4

People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

## shepard_exemplar_similarity_average  — rank 24, 288.4 ± 28.3 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1535.7

People do not infer one concept from the examples jointly; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars (an exemplar-memory account). The similarity of a target to one exemplar is Shepard's consequential-region generalization: how likely a familiar number concept (an arithmetic rule, or a range with a preference for moderate widths) that contains that single exemplar also contains the target. Because each exemplar generalizes on its own and the similarities are averaged rather than combined as joint evidence, adding examples that share a narrow rule (e.g. 4, 16, 12) does not sharpen generalization the way it would for an ideal learner.

## weak_sampling_concept_learning  — rank 25, 288.6 ± 27.3 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1535.8

People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

## range_restricted_rule_concepts  — no comparison row

People do the same Bayesian concept learning as the best model (42 arithmetic rules and every interval of 1-100, a kind-level rule prior, an Erlang width prior favouring decade-bounded ranges, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their hypothesis space also contains range-restricted rules — conjunctions such as "even numbers in the 60s-70s" or "multiples of 4 below 20" — which receive an inferred share of prior belief and, being smaller, are strongly favoured by the size principle when the examples cluster. The sharpest disagreement with the best model lives on rule-consistent targets far from the examples' span (e.g. 12 or 30 after 66,78; 64 or 100 after 4,16,12): the best model says "yes" to every member of the rule wherever it lies, while this model predicts that "yes" to rule members falls off outside the region the examples occupy.

## scale_end_avoiding_ranges  — no comparison row

People do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 exactly as in `family_pooled_rule_prior` (rule-kind prior shares, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but they believe the program's range is unlikely to run all the way to the ends of the 1-100 scale. Each end of a candidate range that sits at 1 or at 100 changes its prior by an inferred factor, so people generalize less toward the extremes of the scale — fewer "yes" answers to numbers in the 90s above high examples (and near 1 below low ones) than a learner who treats the scale's ends like any other number.

## random_example_forgetting  — no comparison row

People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in the best model: kind-level rule prior, size principle, a width prior and a preference for decade-bounded ranges), but their working memory for the examples is unreliable: each example shown is independently lost from memory with an inferred probability (at least one is always retained), and people generalize only from the examples they still hold. So their judgment is an average of what the ideal learner would conclude from each remembered subset of the examples — larger sets are more often reduced to fewer examples, which weakens the size principle's tightening and lets broader concepts and single-example generalizations survive, irrespective of the order the examples were shown in.
