# Refinement menu

The models you may refine, other than the incumbent `upper_edge_marking_ranges`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### prime_concept_excludes_one — rank 1, 0.4 ± 1.8 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -2060.6

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules (kind-level rule prior, Erlang width prior with a decade-boundary preference, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their concept of "prime numbers" is the mathematical one, which excludes 1. The one change is removing 1 from the prime concept: when 1 is among the examples (e.g. 31,3,1) people cannot explain the set as primes and fall back on "odd numbers" (yes to 9 or 25, no special yes to 2), and after a set of primes (e.g. 67,31,17,61) they answer "no" to 1 rather than accepting it as a prime.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/prime_concept_excludes_one.py`

### range_discounted_size_principle — rank 2, 1.0 ± 3.0 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -2061.3

**Hypothesis:** Refinement of `prime_concept_excludes_one`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), ranges of 1-100 and range-restricted rules, but they apply the size principle at full strength only to arithmetic rules and only partly to numeric ranges. The one change: when judging how well a range explains the examples, people treat the range as if it were wider than it is — its effective size lies between its true width and the whole 1-100 scale, by an inferred degree — so more examples tighten range-based generalization less than rule-based generalization, and clustered examples are explained by ranges (with "yes" spreading further across the examples' neighbourhood) more readily than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/range_discounted_size_principle.py`

### odd_composite_concept — rank 3, 1.8 ± 3.4 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -2062.0

**Hypothesis:** People do the same Bayesian concept learning as the best model (arithmetic rules with a kind-level prior, every range of 1-100, range-restricted rules, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their space of concepts also includes a rule with exceptions, "odd numbers that are not prime" (9, 15, 21, 25, 27, 33, ...), as a kind of rule of its own with an inferred prior share. When the examples are all odd composites (e.g. 9,15,25 or 63,27,99,33), this smaller concept wins under the size principle, so people say "yes" to other odd non-primes but "no" to odd primes such as 7, 11 or 53, which "odd numbers" would accept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/odd_composite_concept.py`

### log_odds_yes_bias_concept_learning — rank 4, 1.9 ± 2.7 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2062.2

**Hypothesis:** People infer the program's concept by plain Bayesian concept learning over arithmetic rules and numeric ranges (rules of one kind equally likely, a width prior favouring moderately wide and decade-bounded ranges, the size principle, primes excluding 1), but their answer is not a direct readout of their belief: they carry a constant response bias toward "yes" or toward "no" that shifts their confidence by the same amount on the log-odds scale on every trial. The bias therefore moves answers most for targets the learner is unsure about (posterior near one half) and hardly at all for targets that are clearly in or clearly out of the concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/log_odds_yes_bias_concept_learning.py`

### named_sets_distinct_familiarity — rank 5, 6.5 ± 3.6 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -2066.8

**Hypothesis:** Refinement of `prime_concept_excludes_one`: people do the same Bayesian concept learning (arithmetic rules with a kind-level prior, every range of 1-100 with an Erlang width prior and decade-boundary preference, range-restricted rules, size principle with 10% random examples, primes excluding 1, "yes" = posterior probability the concept contains the target), but the familiar named sets — primes, even numbers, odd numbers, squares and cubes — are not equally likely a priori. The one change: each named set gets its own inferred prior weight instead of sharing one "special sets" weight equally, so how readily people reach for "even/odd" versus "squares" or "primes" (e.g. after 3,7 or 1,4,9 or 31,17) reflects how familiar each concept is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/named_sets_distinct_familiarity.py`

### range_restricted_rule_concepts — rank 6, 49.0 ± 15.0 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2109.2

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 arithmetic rules and every interval of 1-100, a kind-level rule prior, an Erlang width prior favouring decade-bounded ranges, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their hypothesis space also contains range-restricted rules — conjunctions such as "even numbers in the 60s-70s" or "multiples of 4 below 20" — which receive an inferred share of prior belief and, being smaller, are strongly favoured by the size principle when the examples cluster. The sharpest disagreement with the best model lives on rule-consistent targets far from the examples' span (e.g. 12 or 30 after 66,78; 64 or 100 after 4,16,12): the best model says "yes" to every member of the rule wherever it lies, while this model predicts that "yes" to rule members falls off outside the region the examples occupy.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/range_restricted_rule_concepts.py`

### family_pooled_rule_prior — rank 7, 50.2 ± 15.6 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2110.5

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (rule share, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but their prior over rules is set by the kind of rule, not rule by rule. The one change: people hold a belief about how likely each kind of rule is (powers, multiples, final digit, contained digit, and special sets such as primes, evens, odds, squares, cubes) and treat all rules of one kind as equally likely, so "multiples of 7" is as plausible a priori as "multiples of 4", instead of the incumbent's separate free weight for each of the 42 rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/family_pooled_rule_prior.py`

### target_as_weak_evidence — rank 8, 50.4 ± 15.7 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2110.6

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in `family_pooled_rule_prior`: kind-level rule prior, size principle with 10% random examples, Erlang width prior, a preference for decade-bounded ranges), but being asked about a number is itself taken as weak evidence: people partly treat the queried target as if the program had produced it, adding it to the examples with an inferred fraction of an example's weight before judging whether the concept contains it. This suggestion (positive-test) bias inflates "yes" most for targets that fit a small, plausible concept alongside the examples, and least for targets that only a broad or implausible concept would contain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/target_as_weak_evidence.py`

### random_forgetting_range_restricted_rules — rank 9, 50.6 ± 14.9 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2110.8

**Hypothesis:** Refinement of `random_example_forgetting`: people do Bayesian concept learning over 42 arithmetic rules and every range of 1-100 (kind-level rule prior, size principle with 10% random examples, Erlang width prior with a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target) from only the examples they still hold in memory, each shown example being independently forgotten with an inferred probability. The one change, grafted from `range_restricted_rule_concepts`: the concepts people entertain also include range-restricted rules ("even numbers in the 60s", "multiples of 4 below 20") with their own inferred share of prior belief, so the remembered examples can pick out a rule confined to the region they occupy — forgetting loosens generalization on large sets while the conjunctions confine "yes" to rule members near the remembered examples, and the two together predict how far rule-based "yes" extends beyond the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/random_forgetting_range_restricted_rules.py`

### random_example_forgetting — rank 10, 51.5 ± 15.4 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2111.8

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in the best model: kind-level rule prior, size principle, a width prior and a preference for decade-bounded ranges), but their working memory for the examples is unreliable: each example shown is independently lost from memory with an inferred probability (at least one is always retained), and people generalize only from the examples they still hold. So their judgment is an average of what the ideal learner would conclude from each remembered subset of the examples — larger sets are more often reduced to fewer examples, which weakens the size principle's tightening and lets broader concepts and single-example generalizations survive, irrespective of the order the examples were shown in.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/random_example_forgetting.py`

### scale_end_avoiding_ranges — rank 11, 52.5 ± 15.4 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2112.7

**Hypothesis:** People do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 exactly as in `family_pooled_rule_prior` (rule-kind prior shares, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but they believe the program's range is unlikely to run all the way to the ends of the 1-100 scale. Each end of a candidate range that sits at 1 or at 100 changes its prior by an inferred factor, so people generalize less toward the extremes of the scale — fewer "yes" answers to numbers in the 90s above high examples (and near 1 below low ones) than a learner who treats the scale's ends like any other number.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/scale_end_avoiding_ranges.py`

### weber_blurred_ranges_kind_pooled_rules — rank 12, 52.9 ± 15.7 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2113.2

**Hypothesis:** Refinement of `weber_noisy_range_membership`: people do Bayesian concept learning over 42 arithmetic rules and every range of 1-100 (size principle, a prior favouring moderately wide ranges), and they check whether a target falls inside a range on a noisy mental number line whose noise grows with magnitude (Weber's law), so range edges are blurred, more among large numbers, while rule membership is checked exactly. The one change, grafted from `family_pooled_rule_prior`: people's prior over rules is set by the kind of rule (powers, multiples, final digit, contained digit, special sets), with all rules of one kind equally likely, instead of a separate free weight for each of the 42 rules — so the blurred-edge account stops being handicapped by an overfitted rule prior, and range generalization ends in a magnitude-scaled soft edge rather than at decade boundaries.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/weber_blurred_ranges_kind_pooled_rules.py`

### noise_rate_inferring_restricted_rules — rank 13, 56.1 ± 15.3 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2116.4

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, plain ranges of 1-100 and range-restricted rules ("even numbers in the 60s-70s"), with the size principle and "yes" = posterior probability the concept contains the target. The one change is how they treat noisy examples: instead of assuming that exactly 10% of the program's examples are random, people are unsure how many of the examples are random noise and infer the noise rate jointly with the concept for each set, from an inferred prior belief over noise levels. So a set with one stray example (such as 8,24,48,63) can still be read as a tight rule plus one random number, while a set that fits a tight concept exactly drives the belief in noise down and sharpens generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/noise_rate_inferring_restricted_rules.py`

### parity_kind_rule_prior — rank 14, 59.0 ± 15.0 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2119.3

**Hypothesis:** Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target, rules of one kind equally likely a priori), but parity is a kind of rule in its own right. The one change: "even numbers" and "odd numbers" form their own kind with their own inferred prior share, instead of being lumped with primes, squares and cubes as one "special sets" kind, so how readily people fall back on odd/even after examples like 3,7 or 66,78 is set apart from how readily they think of primes, squares or cubes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/parity_kind_rule_prior.py`

### win_stay_lose_shift_concept — rank 15, 101.9 ± 16.0 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2162.2

**Hypothesis:** People do not weigh all the examples jointly; they hold a single running guess about the concept and revise it one example at a time, in the order shown (win-stay, lose-shift). After the first example they adopt one concept drawn in proportion to its posterior plausibility (arithmetic rules with a kind-level prior, ranges with a width prior favouring decade-bounded ranges, size principle); each later example that their current guess contains leaves the guess unchanged, and only an example the guess fails to contain makes them draw a new guess from the posterior given all the examples seen so far. So a broad concept adopted early (e.g. "even numbers" or a wide range) survives later examples that would have narrowed an ideal learner's belief, and the same examples in a different order lead to different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/win_stay_lose_shift_concept.py`

### decade_block_range_concepts — rank 16, 258.0 ± 33.5 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2318.3

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes; rules of one kind equally likely a priori) and numeric ranges, with the size principle, but they only ever think of ranges as whole blocks of decades ("the 80s and 90s", "1 to 30", "41 to 60") — never as a range with arbitrary ends hugging the examples. So range-based generalization is all-or-none within a decade: a target in the same decade block as the examples is accepted as readily as one right next to them, and acceptance drops abruptly at decade boundaries, with no gradient tied to the examples' exact span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/decade_block_range_concepts.py`

### common_feature_contrast_similarity — rank 17, 783.0 ± 40.6 nats behind the best (19.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2843.3

**Hypothesis:** People judge a target by its similarity to the example set as a whole, not by inducing a concept: the psychological similarity is the share of the properties that all the examples have in common (parity, multiples, final digit, contained digit, powers, primes, squares, cubes, decade) that the target also has, each property counted by how distinctive it is (rarer properties count more), together with how close the target is in magnitude to the nearest example on a compressed (logarithmic) distance scale. The probability of "yes" rises smoothly with this similarity, so a target matching the set's distinctive shared properties and lying near an example is accepted, and acceptance falls off gradually — never all-or-none — as it loses shared properties or moves away.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/common_feature_contrast_similarity.py`

## Pruned models (out of the set; narrowest margin first)

### erlang_interval_size_prior — pruned (experiment1 end of experiment)

**Margin:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (size principle, 10% of examples random, "yes" = posterior probability the concept contains the target), but their prior over intervals depends on how wide an interval is rather than where it lies. Following Tenenbaum's original number game, interval width has an Erlang prior with an inferred scale, so people a priori favour moderately sized ranges and disfavour both tiny and near-total ranges, which controls how far they generalize beyond the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_size_prior.py`

### decade_bounded_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `located_sized_interval_prior`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but their prior over where a range lies is not a smooth density over its midpoint: it favours ranges bounded by round decade numbers ("the 20s", "1 to 50", "51 to 100") by an inferred factor. The one change is replacing the Beta-over-midpoint location prior with this round-boundary preference, so generalization from examples tends to stop at decade boundaries — "yes" stays high up to the edge of the decade the examples fall in and drops sharply past it, a sharper edge near the examples' span than a width-only prior predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_bounded_interval_prior.py`

### anchored_endpoint_or_decade_ranges — pruned (experiment1 end of experiment)

**Margin:** 30.6 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `endpoint_marking_interval_sampling`: people do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they assume the program's examples tend to mark the ends of its range — but a range's ends can also be anchored at salient round numbers. The one change is grafting the incumbent's round-decade boundary preference onto the endpoint-marking model: ranges bounded by decade numbers ("the 20s", "1 to 50") get an inferred prior bonus alongside the bonus for ends coinciding with the smallest or largest example, so generalization stops either exactly at the examples' span or at the edge of the enclosing decade, whichever anchors are available.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/anchored_endpoint_or_decade_ranges.py`

### endpoint_marking_interval_sampling — pruned (experiment1 end of experiment)

**Margin:** 30.8 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but when two or more distinct examples are shown they assume the program's examples tend to mark the ends of its range (edge-marking, pedagogical sampling). The one change is an inferred evidence bonus for each end of a candidate range that coincides with the smallest or largest example, so the range that exactly spans the examples is favoured over wider ones, and people say "yes" inside the examples' span but sharply less just beyond it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/endpoint_marking_interval_sampling.py`

### erlang_interval_response_lapse — pruned (experiment1 end of experiment)

**Margin:** 31.3 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_response_lapse.py`

### weber_noisy_range_membership — pruned (experiment1 end of experiment)

**Margin:** 31.8 nats behind scale_end_avoiding_ranges (4.2× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (size principle, a prior favouring moderately wide ranges), but they check whether a target falls inside a range on a noisy analog mental number line whose noise grows in proportion to magnitude (Weber's law), while rule membership (even, multiple of 5, ...) is checked exactly. So the edge of a range is blurred rather than sharp: targets just outside the range sometimes count as "in" and targets just inside sometimes as "out", and this blur is wider among large numbers than among small ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weber_noisy_range_membership.py`

### gamma_shape_interval_width_prior — pruned (experiment1 end of experiment)

**Margin:** 32.5 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/gamma_shape_interval_width_prior.py`

### primacy_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 32.7 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/primacy_weighted_evidence.py`

### few_sample_majority_vote — pruned (experiment1 end of experiment)

**Margin:** 32.9 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/few_sample_majority_vote.py`

### recency_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 33.4 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_evidence.py`

### per_endpoint_round_boundary_prior — pruned (experiment1 end of experiment)

**Margin:** 33.5 nats behind scale_end_avoiding_ranges (4.4× dse)

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) and they favour ranges with round boundaries, but they judge roundness one endpoint at a time and under either everyday convention. The one change: each end of a range that falls on a round decade number — a lower end like 1, 10, 11, 20, 21, ... or an upper end like 9, 10, 19, 20, ... 100 (so "the 20s" as 20-29, 21-30 or 20-30) — multiplies the range's prior by an inferred factor, so a range with one round end and one end hugging the examples is also favoured, instead of only ranges whose both ends sit at the incumbent's strict 21-30 convention.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/per_endpoint_round_boundary_prior.py`

### offset_arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 34.5 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning as in the best model (42 arithmetic rules and every interval of 1-100, with an Erlang width prior and a preference for decade-bounded ranges, size principle, "yes" = posterior probability the concept contains the target), but their space of candidate rules also includes offset arithmetic progressions — "every 4th number starting from 3" (3, 7, 11, ...), "every 6th number from 5" (5, 11, ..., 77, 83) — for every step from 3 to 12 and every non-zero starting offset. These progressions together receive an inferred share of prior belief, split equally among them, so examples whose gaps share a common step lead people to say "yes" to targets that continue the same counting pattern, even when they fall outside the examples' range and no multiple, digit or interval rule covers them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_arithmetic_progression_concepts.py`

### located_sized_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 35.8 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/located_sized_interval_prior.py`

### noisy_criterion_threshold — pruned (experiment1 end of experiment)

**Margin:** 38.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noisy_criterion_threshold.py`

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Margin:** 39.3 nats behind scale_end_avoiding_ranges (4.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_interval.py`

### tempered_evidence_interval — pruned (experiment1 end of experiment)

**Margin:** 41.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_interval.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 42.5 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### noise_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 43.9 nats behind scale_end_avoiding_ranges (5.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noise_tolerant_concept_learning.py`

### log_scale_erlang_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 53.4 nats behind scale_end_avoiding_ranges (5.4× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_scale_erlang_interval_prior.py`

### minimal_consistent_concept — pruned (experiment1 end of experiment)

**Margin:** 240.9 nats behind scale_end_avoiding_ranges (10.8× dse)

**Hypothesis:** People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/minimal_consistent_concept.py`

### arithmetic_step_lattice — pruned (experiment1 end of experiment)

**Margin:** 251.6 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_step_lattice.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 268.5 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### span_scaled_hull_similarity — pruned (experiment1 end of experiment)

**Margin:** 283.2 nats behind scale_end_avoiding_ranges (11.1× dse)

**Hypothesis:** People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/span_scaled_hull_similarity.py`

### shepard_exemplar_similarity_average — pruned (experiment1 end of experiment)

**Margin:** 289.5 nats behind scale_end_avoiding_ranges (10.3× dse)

**Hypothesis:** People do not infer one concept from the examples jointly; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars (an exemplar-memory account). The similarity of a target to one exemplar is Shepard's consequential-region generalization: how likely a familiar number concept (an arithmetic rule, or a range with a preference for moderate widths) that contains that single exemplar also contains the target. Because each exemplar generalizes on its own and the similarities are averaged rather than combined as joint evidence, adding examples that share a narrow rule (e.g. 4, 16, 12) does not sharpen generalization the way it would for an ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shepard_exemplar_similarity_average.py`

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 289.6 nats behind scale_end_avoiding_ranges (10.7× dse)

**Hypothesis:** People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_learning.py`
