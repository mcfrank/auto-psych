# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **graded_exemplar_equal_rule_prior** (posterior=0.476, elpd_loo=-2035.53)
- Trials: 4500
- Models compared: 5

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| graded_exemplar_equal_rule_prior | 0.4755 | -2035.53 |
| prior_matched_familiar_strays | 0.3471 | -2035.85 |
| two_source_program_concepts | 0.0996 | -2037.10 |
| familiar_strays_equal_rule_prior | 0.0476 | -2037.83 |
| rule_count_familiar_strays | 0.0302 | -2038.29 |

## Hypotheses

- **graded_exemplar_equal_rule_prior**: Refinement of `graded_exemplar_familiar_strays`: people hold familiar number concepts (mathematical rules and magnitude ranges of 1-100) as stored member lists, retrieve a concept to the extent that the examples resemble its members on the number line, allow for an uncertain rate of stray examples that are sometimes familiar numbers, and endorse a target by its closeness to the nearest stored member of the retrieved concepts. The one change, taken from `familiar_strays_equal_rule_prior`, is the rule prior: people have no preference among kinds of rule at all (parity, primes, squares, cubes, powers, multiples, final digit, contained digit are equally plausible a priori), so which rule is retrieved is decided only by how closely and how tightly its members match the examples.
- **prior_matched_familiar_strays**: Refinement of `familiar_strays_equal_rule_prior`: people do the same Bayesian concept learning over equally plausible mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes "familiar" numbers (a member of some well-known rule). The one change is where their idea of a stray comes from. A stray is simply the output of some other program drawn from their own prior over programs: with the same probability they give a rule-based program a priori, it is a member of a randomly chosen rule, and otherwise it is any number at all. So how often strays are familiar numbers is not a separate free belief. It is fixed by how strongly people expect rule-based programs to begin with, and someone who favours rules more also writes off familiar-looking outliers more readily.
- **two_source_program_concepts**: People think the hidden program may run two sources at once: a main concept (a mathematical rule or a magnitude range of 1-100, learned with the size principle and an uncertain belief about how many examples come from it) and a secondary source that produced the off-pattern examples, which is either random glitches or a second familiar rule. Examples that break the main concept are therefore not just written off: if they fit a small familiar rule (e.g. 64 among 13, 23, 43), people infer the program also uses that rule and say "yes" to its other members (32, 8), whereas off-pattern examples that fit no small rule are treated as glitches that license no further generalisation.
- **familiar_strays_equal_rule_prior**: Refinement of `familiar_strays_family_level_prior`: people do the same Bayesian concept learning over mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes "familiar" numbers (a member of some well-known rule) rather than random ones. The one change is the rule prior: people have no preference among kinds of rule at all — parity, primes, squares, cubes, powers, multiples, final digit and contained digit rules are all equally plausible a priori — so which rule they adopt is decided only by how many examples it explains and how small it is (the size principle), and the rules they would generate a familiar stray from are the same equally weighted rules.
- **rule_count_familiar_strays**: Refinement of `familiar_strays_equal_rule_prior`: people do the same Bayesian concept learning over equally plausible mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes "familiar" numbers rather than random ones. The one change is what makes a number familiar. It is not how readily a random member of a random rule lands on it, which favours members of tiny rules such as 27, 64 and 81. It is how many well-known rules the number belongs to, its breadth of mathematical associations. So richly connected numbers (12, 24, 36, 60, 100) are the easiest examples to write off as strays, while a lone member of a tiny rule is no more excusable than other numbers with as many associations. This adds no free parameter.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| graded_exemplar_equal_rule_prior ←selected | 0.00 | 0.00 | — (best) | 0.476 | yes |
| prior_matched_familiar_strays | 0.31 | 2.71 | no (within ~2·dse) | 0.347 | yes |
| two_source_program_concepts | 1.56 | 3.60 | no (within ~2·dse) | 0.100 | yes |
| familiar_strays_equal_rule_prior | 2.30 | 2.45 | no (within ~2·dse) | 0.048 | yes |
| rule_count_familiar_strays | 2.76 | 2.79 | no (within ~2·dse) | 0.030 | yes |
