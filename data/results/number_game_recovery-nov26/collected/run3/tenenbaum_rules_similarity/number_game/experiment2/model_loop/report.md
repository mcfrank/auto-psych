# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **mixed_strong_weak_sampling_concepts** (posterior=0.416, elpd_loo=-2283.94)
- Trials: 4500
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| mixed_strong_weak_sampling_concepts | 0.4160 | -2283.94 |
| range_size_misperception_concepts | 0.1655 | -2284.87 |
| beta_share_width_exception_concepts | 0.1206 | -2285.18 |
| specificity_biased_rule_prior | 0.0829 | -2285.56 |
| uncertainty_shifted_criterion | 0.0657 | -2285.79 |
| boundary_examples_anchor_ranges | 0.0535 | -2285.99 |
| mixed_sampling_prior_leaking_judgment | 0.0529 | -2286.01 |
| individual_rule_vs_range_tendency | 0.0429 | -2286.22 |

## Hypotheses

- **mixed_strong_weak_sampling_concepts**: Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples. The one change is their assumption about how the examples were generated: they believe only some genuine examples (an inferred share) were drawn at random from the concept, which is what drives the size principle, while the rest were merely numbers the concept happens to contain, chosen without regard to how big the concept is (mixed strong/weak sampling). So the size principle is diluted most for big concepts, and after examples that fit both a narrow and a broad concept (multiples of 9 vs multiples of 3, a tight range vs a wide one) people keep the broad concept in play and say "yes" to its other members more than pure random sampling predicts.
- **range_size_misperception_concepts**: People do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they misjudge how many numbers a range holds: a range's felt size grows as an inferred power of its true width, so (if the power is below one) wide ranges seem to hold fewer numbers than they do and are penalised too little by the size principle — or, above one, too much. Because this distortion acts on the evidence each example gives rather than on the prior, it shows up most with more examples: with a compressed sense of range size, adding examples inside a span narrows magnitude generalisation less than the normative learner predicts, so "yes" extends further beyond the examples' span for three- and four-example sets.
- **beta_share_width_exception_concepts**: Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges, with the size principle. The one change is the shape of their prior about how wide a range is: they think of a range as a portion of the bounded 1–100 line and expect it to cover a typical share of it (a beta prior over that share, with its mean and sharpness inferred), rather than holding an unbounded gamma expectation over raw width. So ranges covering nearly the whole line are as implausible to them as very narrow ones, and when the examples are spread widely (e.g. 43, 4, 91, 9) people fall back on a rule instead of an "almost any number" range. As a result they say "yes" less often to numbers that only such a near-total range would include.
- **specificity_biased_rule_prior**: People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but within each kind of rule they do not find all rules equally plausible beforehand: a rule's prior plausibility scales with how specific it is (how few numbers it contains), to an inferred degree, so "multiples of 9" is a priori favoured over "multiples of 3" and "multiples of 8" over "multiples of 4" (or disfavoured, if the degree is negative). The sharpest disagreement with the best model lives in example sets that fit a sparse rule nested inside a broader rule of the same kind, especially with only two examples (e.g. 81, 9; 8, 32; 6, 66): there this model confines "yes" to multiples of the larger base and rejects the other multiples of its divisor far more firmly than the size principle alone does.
- **uncertainty_shifted_criterion**: People infer the hidden concept exactly as the best model's Bayesian concept learner does, but their yes/no decision criterion depends on how sure they are which concept the program uses: the more their belief is spread over many competing concepts (rules and range widths), the more they shift their answer away from the inferred membership probability toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.
- **boundary_examples_anchor_ranges**: When people consider whether the examples come from a magnitude range, they judge the range mainly by the smallest and largest examples, which mark its edges. The examples in between tell them nothing new about where the edges are, so they count for less (to an inferred degree) as evidence that the range is narrow, while every example counts fully when they test an arithmetic rule. As a result, with three or four examples people keep entertaining wider ranges, and a range explanation stays more competitive with a rule, than an ideal learner who weighs every example equally would.
- **mixed_sampling_prior_leaking_judgment**: Refinement of `mixed_strong_weak_sampling_concepts` with one component taken from `beta_share_width_prior_leaking_judgment`: people infer the hidden concept exactly as the incumbent's learner does (family-organised rules with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, occasional stray examples, and only an inferred share of examples taken as random draws that drive the size principle). The one change is in the yes/no judgment: it is pulled by how plausible the target was before any examples were shown, so the log-odds of saying "yes" is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership. A priori typical numbers (mid-range numbers that most moderately wide ranges contain, members of many rules) therefore get more "yes" than the examples warrant, or less if the share is negative, on top of the broad-concept generalisation that mixed sampling produces.
- **individual_rule_vs_range_tendency**: Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is that people differ, stably across their session, in how readily they take the hidden program to follow an arithmetic rule rather than a magnitude range: each person has their own prior rule-versus-range odds, drawn from a population distribution whose mean and spread are inferred. For the same examples, a "rule-minded" person confines "yes" to members of the fitting rule while a "magnitude-minded" person spreads "yes" over nearby numbers, so their answers to the same (set, target) pair disagree systematically rather than at random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| mixed_strong_weak_sampling_concepts ←selected | 0.00 | 0.00 | — (best) | 0.416 | yes |
| range_size_misperception_concepts | 0.92 | 1.57 | no (within ~2·dse) | 0.166 | yes |
| beta_share_width_exception_concepts | 1.24 | 1.11 | no (within ~2·dse) | 0.121 | yes |
| specificity_biased_rule_prior | 1.61 | 1.41 | no (within ~2·dse) | 0.083 | yes |
| uncertainty_shifted_criterion | 1.85 | 1.52 | no (within ~2·dse) | 0.066 | yes |
| boundary_examples_anchor_ranges | 2.05 | 1.68 | no (within ~2·dse) | 0.054 | yes |
| mixed_sampling_prior_leaking_judgment | 2.06 | 1.72 | no (within ~2·dse) | 0.053 | yes |
| individual_rule_vs_range_tendency | 2.27 | 1.21 | no (within ~2·dse) | 0.043 | yes |
