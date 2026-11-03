# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **representative_coverage_family_doubt** (posterior=0.353, elpd_loo=-3662.11)
- Trials: 6750
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| representative_coverage_family_doubt | 0.3530 | -3662.11 |
| rule_family_doubt_criterion | 0.1535 | -3662.94 |
| kind_uncertainty_shifted_criterion | 0.0909 | -3663.46 |
| mixed_sampling_rule_family_doubt | 0.0874 | -3663.50 |
| representative_coverage_sampling | 0.0872 | -3663.51 |
| two_cluster_rule_family_doubt | 0.0856 | -3663.52 |
| uncertainty_shifted_criterion | 0.0787 | -3663.61 |
| first_impression_kind_anchoring | 0.0637 | -3663.82 |

## Hypotheses

- **representative_coverage_family_doubt**: Refinement of the live model `representative_coverage_sampling`, with one component taken from the incumbent `rule_family_doubt_criterion`. People infer the hidden concept as a Bayesian concept learner does but assume the examples are representative of the program's output, so they discount (or, if the inferred degree is negative, favour) concepts most of whose members lie outside the stretch the examples cover. The one change is the yes/no criterion: as in the incumbent, whatever doubt remains about which kind of arithmetic rule the program follows (weighted by how much they believe it is a rule at all) shifts their answer toward "no" (or "yes"), so the representativeness judgment both changes which concepts they generalise from and how much rule-kind doubt is left to move the criterion.
- **rule_family_doubt_criterion**: Refinement of `kind_uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are about what kind of program they face. The one change is which part of that kind-level doubt moves the criterion: only doubt about which kind of arithmetic rule it is (parity, primes, powers, multiples, final digit, ...), weighted by how much they believe it is a rule at all, shifts answers toward "no" (or "yes"), while doubt about whether the program is a rule or a stretch of the number line does not — the complement of `rule_vs_range_doubt_criterion`, which kept only that rule-versus-range part.
- **kind_uncertainty_shifted_criterion**: Refinement of `uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are, but the one change is what that uncertainty is about. It is not their spread of belief over individual concepts (which rule, which range width) but their uncertainty about what kind of program they face — a parity rule, a prime rule, a power rule, a multiples rule, a digit rule, or a magnitude range — so examples that leave it open whether the program follows an arithmetic rule or a stretch of the number line move the answer toward "no" (or "yes"), while examples that clearly point to one kind do not, however many concepts of that kind remain open.
- **mixed_sampling_rule_family_doubt**: Refinement of the incumbent `rule_family_doubt_criterion`, with one component taken from `mixed_strong_weak_sampling_concepts`: people infer the hidden concept as the incumbent's Bayesian learner does and shift their yes/no criterion with their doubt about which kind of arithmetic rule the program follows (weighted by how much they believe it is a rule at all). The one change is their sampling assumption: they take only an inferred share of the examples as random draws from the concept (the size principle) and the rest as mere members of it, so after examples that fit a narrow rule, broad rules of other kinds (e.g. all even numbers alongside powers of 2) stay in play — broadening "yes" to those rules' other members and raising exactly the rule-kind doubt that shifts the criterion for those sets.
- **representative_coverage_sampling**: People assume the examples they are shown are representative of the hidden program's output: a fair sample spreads out over the concept rather than bunching in one corner of it. So, beyond the size principle, they discount a candidate concept (rule or range) to an inferred degree when most of its members lie outside the stretch the examples cover (e.g. "all even numbers" after 2, 4, 6, 8) and favour concepts the examples span well (e.g. powers of 2 after 2, 8, 16, 64), and they say "yes" to the target by how strongly the concepts they then believe contain it.
- **two_cluster_rule_family_doubt**: Refinement of `two_cluster_range_concepts`, with one component taken from the incumbent `rule_family_doubt_criterion`. People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), single magnitude ranges and "two separate groups" concepts (two ranges with a gap between them), all judged as a typical share of the 1–100 line and with the size principle. The one change is the yes/no criterion: only doubt about which kind of arithmetic rule the program follows, weighted by how much they believe it is a rule at all, shifts the answer from the inferred membership probability toward "no" (or "yes"), while doubt about whether the numbers form one stretch, two groups or a rule does not — so for widely split sets the two-group concept both carries generalisation away from the empty middle and, by drawing belief away from rules, weakens the criterion shift there.
- **uncertainty_shifted_criterion**: People infer the hidden concept exactly as the best model's Bayesian concept learner does, but their yes/no decision criterion depends on how sure they are which concept the program uses: the more their belief is spread over many competing concepts (rules and range widths), the more they shift their answer away from the inferred membership probability toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.
- **first_impression_kind_anchoring**: People decide what kind of program they face (a parity, prime, square/cube, power, multiples, final-digit, contained-digit or exception rule, or a stretch of the number line) largely from their first impression of the first example, and later examples only partly adjust that impression (anchoring and adjustment), while which particular concept of the favoured kind it is gets worked out from all the examples. So the same examples in a different order lead to different answers, and a set whose first example suggests a different kind than the whole set does (e.g. 50 then 2, 8, 64) keeps the first-suggested kind in play, giving "yes" to targets that only that kind contains more often than a learner who weighs all examples equally would; otherwise they answer as the best model does, including its shift of criterion with doubt about which kind of rule it is.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| representative_coverage_family_doubt ←selected | 0.00 | 0.00 | — (best) | 0.353 | yes |
| rule_family_doubt_criterion | 0.83 | 1.30 | no (within ~2·dse) | 0.154 | yes |
| kind_uncertainty_shifted_criterion | 1.36 | 1.77 | no (within ~2·dse) | 0.091 | yes |
| mixed_sampling_rule_family_doubt | 1.40 | 1.73 | no (within ~2·dse) | 0.087 | yes |
| representative_coverage_sampling | 1.40 | 1.48 | no (within ~2·dse) | 0.087 | yes |
| two_cluster_rule_family_doubt | 1.42 | 1.49 | no (within ~2·dse) | 0.086 | yes |
| uncertainty_shifted_criterion | 1.50 | 2.03 | no (within ~2·dse) | 0.079 | yes |
| first_impression_kind_anchoring | 1.71 | 2.22 | no (within ~2·dse) | 0.064 | yes |
