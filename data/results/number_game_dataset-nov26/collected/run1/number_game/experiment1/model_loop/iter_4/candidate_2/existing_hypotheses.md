# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## shared_factor_parity_exemplar  — rank 0, the best model on this data, ELPD-LOO -1576.6

Refinement of `variability_tuned_parity_attention`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity, with attention to each set by how variable the examples are on it. The one change is a further, fixed "shared factor" dimension: a target that has no common divisor with an example (they are coprime) loses similarity to that example by a constant factor. So after "3, 63" people endorse 36 and 39 (they share the factor 3) although these match neither example's final digit nor magnitude, and after "8, 32" they reject 37, unlike targets that share a factor with the examples.

## divisibility_kinship_exemplar  — rank 1, 1.6 ± 1.1 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1578.2

Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples. Attention to magnitude and final digit is still set by how variable the examples are, and parity is still a fixed dimension. The one change is that the parity dimension is widened into divisibility kinship: an example also contributes less similarity when it and the target have no common factor at all (they are coprime), so two numbers that share a factor such as 3, 5 or 7 feel related even when their parity differs. This predicts that after sets like "3, 63", "6, 66" or "55, 15, 75, 45", targets that share a factor with the examples (30, 36, 39, 60, 50) are endorsed well above what magnitude, final digit and parity alone allow, while targets coprime to the examples (e.g. 37 after "8, 32") are rejected more often.

## digit_run_strict_criterion_exemplar  — rank 2, 3.0 ± 9.5 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -1579.6

Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, plus a fixed parity penalty), but how much similarity is enough to say "yes" depends on the set. The one change: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and raise their response criterion for every target, so such sets are endorsed less overall, and magnitude-near targets with the wrong digit are rejected especially often, than similarity alone predicts. Other sets keep the shared criterion.

## shared_regularity_exemplar  — rank 3, 7.1 ± 5.8 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1583.7

Besides generalizing by similarity to the remembered examples (magnitude, final digit and parity, with attention to magnitude and final digit tuned by how variable the examples are), people notice the explicit arithmetic and digit regularities that all of the examples share — multiples of k, powers, squares, cubes, primes, odd/even, ending in a given digit — and weigh them by how surprising they are (the fewer numbers a regularity covers, the more it counts). A target that conforms to the most specific shared regularity is endorsed more, and one that breaks the most specific shared regularity is rejected more, so rule-coherent sets (e.g. "8, 32", "78, 48, 88", "3, 63") generalize to rule members far from every example and reject close non-members, making sets differ more in endorsement than similarity alone allows.

## graded_parity_variability_exemplar  — rank 4, 8.3 ± 4.4 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1584.9

Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change, grafted from `parity_place_value_exemplar_similarity`, is that parity (odd/even) is a fixed, always-attended dimension of the similarity space, not one whose weight depends on the set: every example of the opposite parity to the target contributes less similarity by the same factor, so final digits are compared in a graded way (same digit > same parity > opposite parity). This predicts that same-parity targets far from every example and sharing no final digit are endorsed more after same-parity sets, and also that in mixed-parity sets a target is generalized mostly from its same-parity examples rather than parity being ignored.

## variability_tuned_parity_attention  — rank 5, 8.6 ± 4.3 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -1585.2

People judge a target by its summed similarity to the remembered examples in a psychological space of magnitude, final digit and parity (odd/even), and they set their attention to every one of these dimensions from how variable the examples are on it: scattered examples make magnitude generalization broad, a shared final digit makes a final-digit mismatch heavily penalized, and examples that are all even (or all odd) make an opposite-parity target feel dissimilar. So after an all-even set, even targets far from every example and sharing no final digit with them are still endorsed more than odd ones, while after mixed-parity sets parity plays almost no role.

## parity_tuned_exemplar_attention  — rank 6, 9.4 ± 4.4 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1585.9

Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change is that parity (odd vs even) becomes a further dimension of the same kind: the more uniformly the examples share parity (e.g. all even), the more a target of the opposite parity loses similarity, so after same-parity sets people endorse same-parity targets even far from every example and sharing no final digit, while mixed-parity sets give parity no weight.

## parity_space_digit_tuned_exemplar  — rank 7, 13.3 ± 6.6 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1589.9

Refinement of `parity_place_value_exemplar_similarity` (one graft from `digit_consistency_attention_exemplar`): people judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity (odd/even), but the attention they pay to the final digit adapts to the examples — the more consistently the examples share a final digit, the more a final-digit mismatch reduces similarity. The parity dimension lets same-parity targets far from every example still feel like they belong (the parity contrast the best model misses), while the adaptive digit attention makes digit-coherent sets like "46, 76, 66" generalize narrowly.

## variability_tuned_exemplar_attention  — rank 8, 17.2 ± 7.0 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1593.8

People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but they set their attention to each dimension from how variable the examples are on it: widely spread examples make them generalize broadly in magnitude while tightly clustered examples make magnitude matter sharply, and examples that share a final digit make a final-digit mismatch heavily penalized. So the same target can be endorsed after a scattered set and rejected after a homogeneous one, making sets differ strongly in overall endorsement.

## serial_position_weighted_exemplar  — rank 9, 25.6 ± 7.8 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1602.2

People judge a target by its summed similarity to the remembered examples (on magnitude and final digit, with attention tuned to how variable the examples are), but the examples are not remembered equally: memory strength changes steadily with serial position, so either the most recent examples (recency) or the first ones (primacy) dominate the judgment. A target close to the last-shown example is therefore endorsed more (under recency) than an equally close target near the first-shown example, with the direction and strength of this order effect inferred from the data.

## digit_consistency_attention_exemplar  — rank 10, 28.7 ± 9.1 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1605.2

Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude and final digit, but how much attention they pay to the final digit adapts to the examples shown. The one change is that the more consistently the examples share a final digit (e.g. "46, 76, 66"), the more heavily a final-digit mismatch reduces similarity, so such sets produce narrow, digit-bound generalization, while sets with varied final digits are generalized mainly by magnitude.

## coherence_driven_digit_attention  — rank 11, 30.2 ± 9.1 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1606.8

People judge a target by its summed similarity to the remembered examples (magnitude and final digit), but how much attention they give to the final digit is set by reading the examples: the more the examples agree with one another on their final digit, the more attention shifts onto it. So a set like "15, 85, 45" makes people say yes almost only to numbers ending in 5, while a set with mixed final digits makes them generalize broadly by magnitude. This makes overall endorsement differ strongly between sets.

## parity_place_value_exemplar_similarity  — rank 12, 39.7 ± 10.8 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1616.3

Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples, but the psychological space in which numbers are compared has a third dimension, parity — similarity to an example drops by an additional fixed factor when the target and that example differ in being odd or even. The single change is this parity dimension, so after mostly-even (or mostly-odd) examples, same-parity targets far from every example still feel like they "belong", which the magnitude-and-final-digit space cannot produce.

## digit_consistency_attention_exemplar_2  — rank 13, 40.6 ± 10.7 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1617.2

Refinement of `exemplar_magnitude_digit_similarity`: people still judge a target by its summed similarity to the remembered examples on magnitude and final digit, and say "yes" by a Luce ratio of that similarity to a fixed criterion. The one change is in attention: how much weight people give the final digit depends on how consistently the examples share final digits. When the examples agree on their last digit (e.g. 78, 48, 88), a mismatching last digit cuts similarity much more sharply. So these sets generalize narrowly to numbers with the shared digit, while sets whose last digits vary generalize mostly by magnitude.

## cohesion_referenced_exemplar  — rank 14, 43.2 ± 9.7 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1619.8

People judge a target by its similarity to the remembered examples (on magnitude, final digit and parity), but they do not compare that similarity to a fixed standard: they compare it to how similar the examples are to one another, asking "does this number fit in with the examples as well as they fit with each other?". So a tightly coherent set (e.g. 46, 76, 66, or 12, 14, 16) sets a high bar and yields narrow generalization and low overall endorsement, while a scattered set sets a low bar and lets many targets in, making overall willingness to say yes vary strongly from set to set.

## similarity_judgment_with_random_lapses  — rank 15, 43.7 ± 11.7 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1620.3

People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but on a fraction of trials they lapse — they do not consult the examples at all and simply guess yes or no at random (a coin flip). So even targets identical or very close to an example sometimes get "no", and targets dissimilar to every example sometimes get "yes", at a floor and ceiling set by one lapse rate rather than by similarity.

## feature_distribution_prototype  — rank 16, 43.7 ± 12.5 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1620.3

People do not store examples or test rules; they summarise the examples into a category description — an estimated bell-shaped distribution of magnitudes (centred on the examples' mean, as wide as their spread plus some prior uncertainty) and estimated frequencies of final digits and of odd/even, smoothed toward chance — and they say yes in proportion to how much more probable the target is under that category description than under a random number. Because tight sets yield a narrow, peaked category and digit-coherent sets yield a sharp digit profile, sets differ strongly in how readily any target is accepted, and a magnitude-near target with the wrong final digit is rejected after a digit-homogeneous set.

## place_value_exemplar_similarity  — rank 17, 47.6 ± 11.8 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1624.2

People do not induce a rule; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers is computed in a two-dimensional psychological space of overall magnitude and final digit. Similarity decays exponentially with the numeric distance between the numbers and drops by a fixed factor when their final digits differ, so targets near the examples or sharing an example's last digit feel like they "belong", and the more total similarity a target accrues the more likely the answer is yes.

## exemplar_magnitude_digit_similarity  — rank 18, 52.3 ± 12.4 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1628.9

People do not reason about hidden rules at all; they judge a target by its summed similarity to the remembered example numbers (an exemplar / GCM-style process). Each number is represented psychologically by its magnitude and by its final digit, similarity to an example decays exponentially with attention-weighted city-block distance on those two dimensions, and the probability of saying "yes" is the target's summed similarity relative to a fixed response criterion (Luce choice rule).

## digit_numeral_exemplar_similarity  — rank 19, 79.0 ± 15.1 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1655.5

People do not induce a rule or interval at all; they judge the target by its summed similarity to the examples treated as two-digit numerals, where similarity to an example falls off with the distance between tens digits and drops by a fixed factor when the final digits differ. This predicts partial generalization by shared final digit even in mixed sets (e.g. 72 or 12 after "92,82,27,42") and graded interpolation through neighbouring decades, and disagrees most sharply with the rule-based Bayesian model on sets that form a mathematical rule (e.g. powers of 2 or multiples of 7), where it predicts no special endorsement of rule members that share neither tens nor final digits with any example.

## tempered_size_principle_mixture  — rank 20, 128.7 ± 24.0 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1705.3

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals of 1-100, but they apply the size principle only partially — the evidence each example provides is tempered by a single inferred exponent, so smaller concepts are favoured less steeply than strong sampling implies and people hedge between broad and narrow concepts. The one change is this free size-principle strength (an exponent on the example likelihood, applied identically to rules and intervals); "yes" is still the posterior probability that the concept contains the target.

## tempered_size_principle_concepts  — rank 21, 137.9 ± 27.4 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1714.4

Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 45 rules and all intervals of 1-100 with independently weighted rules, but people apply a weakened (or strengthened) size principle — the likelihood of an example inside a concept falls with the concept's size raised to a fitted exponent rather than exactly one over its size. The single change is that exponent; a value below one lets broad intervals spanning scattered examples compete with small coincidental rules, producing more generalization within the examples' range.

## memory_limited_concept_learning  — rank 22, 147.4 ± 25.4 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1723.9

People do Bayesian concept learning over rules (odd, even, primes, multiples, powers, final digits, digits contained) and intervals of 1-100 with the size principle, but under a working-memory limit: each example shown is retained only with some probability, and the learner reasons as if the forgotten examples were never seen. A judgment is therefore an average over the concepts implied by the remembered subsets of examples, so a single stray example (e.g. 27 among 92, 82, 42) often fails to veto a coherent rule, and sets differ sharply in endorsement depending on how robustly their examples point to one concept.

## bigelow_piantadosi_independent  — rank 23, 201.7 ± 37.5 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1778.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## interval_width_prior_concepts  — rank 24, 213.6 ± 37.5 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1790.1

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size-principle sampling with 10% noise, "yes" = posterior probability the concept contains the target), but the one change is that intervals are not equally likely a priori — an interval's prior weight decays exponentially with its width at a rate inferred from the data, so people favour tight ranges hugging the examples and generalize more sharply to numbers inside the example span than to those outside it.

## bigelow_piantadosi_mixture  — rank 25, 265.7 ± 40.3 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1842.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 26, 479.3 ± 53.8 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2055.9

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
