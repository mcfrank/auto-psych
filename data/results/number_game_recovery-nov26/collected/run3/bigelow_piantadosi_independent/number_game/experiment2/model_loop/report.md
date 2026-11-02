# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **irregular_threes_times_table** (posterior=0.857, elpd_loo=-2198.65)
- Trials: 4500
- Models compared: 6

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| irregular_threes_times_table | 0.8570 | -2198.65 |
| prime_over_odd_reading | 0.1307 | -2200.54 |
| repdigit_pattern_chart_concepts | 0.0073 | -2203.42 |
| prime_salient_marked_odd | 0.0044 | -2203.93 |
| hidden_divisibility_by_three | 0.0006 | -2205.88 |
| prime_salient_hundred_chart | 0.0000 | -2210.89 |

## Hypotheses

- **irregular_threes_times_table**: Refinement of the incumbent `prime_over_odd_reading`. Everything in it is kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, the single preference that moves plausibility from "odd" to "prime", Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change is that the 3 times table has no visible pattern in the written numeral (multiples of 2, 5 or 10 can be read off the last digit, multiples of 3 cannot), so people seldom notice that a set is all "multiples of 3", and that one rule carries its own fitted extra prior cost. So after sets of multiples of 3 such as 69, 9, 39, 21 or 63, 27, 99, 33, people extend "odd", a final digit or a range instead, and say yes to other multiples of 3 hardly more often than to non-multiples, where the incumbent over-generalises "multiples of 3".
- **prime_over_odd_reading**: Refinement of `hundred_chart_row_ranges` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment). The one change is that "odd numbers" and "prime numbers" are rival readings of the same kind of evidence (numbers with no factor of 2), and people find "prime" the more meaningful reading and "odd" the emptier, merely "not even" one: a single fitted preference moves prior plausibility away from "odd numbers" and onto "prime numbers" by the same amount. So after all-prime sets (2, 47; 53, 2; 67, 31, 17, 61) people say yes to other primes and no to odd composites, and after all-odd non-prime sets (69, 9, 39, 21) they extend "odd" less than the incumbent predicts and fall back on ranges or narrower rules.
- **repdigit_pattern_chart_concepts**: Refinement of the incumbent `prime_over_odd_reading`, with everything in it kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, the single preference that moves plausibility from "odd" to "prime", Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change adds a concept the incumbent lacks. People notice when a numeral's two digits are the same (11, 22, ..., 99), so "repeated-digit numbers" is a candidate rule, priced like other two-primitive rules plus its own fitted prior bonus. After sets holding two or more repdigits (e.g. 33, 77 or 44, 11, 88), people say yes to other repdigits and no to numbers that merely lie near the examples, which closes the critique's report that the incumbent under-predicts yes to repdigit targets after repdigit sets.
- **prime_salient_marked_odd**: Refinement of `prime_salient_hundred_chart` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, a fitted prior bonus for the familiar category "prime numbers", the size principle and near-all-or-none commitment). The one change, taken from `marked_odd_parity_concepts`, is that "odd" is a marked, negatively defined category ("not even"), so the rule "odd numbers" carries a fitted extra prior cost that "even numbers" does not. As a result, after all-prime sets of odd primes (53, 47, 41; 67, 31, 17, 61) "odd" competes less with "primes", and people say yes to other primes but no to odd composites (9, 21, 63, 81) more sharply than prime salience alone predicts, while after all-even sets they still extend "even" readily.
- **hidden_divisibility_by_three**: Refinement of `prime_salient_marked_odd` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, a prime bonus, a marked-odd cost, the size principle and near-all-or-none commitment). The one change is that divisibility by 3 cannot be read off a numeral's last digit the way divisibility by 2, 5 or 10 can; it needs a digit-sum check. So every rule whose membership test contains that check ("multiples of 3", "of 6", "of 9", "of 12") carries one shared, fitted extra prior cost. As a result, after odd multiples of 3 such as 69, 9, 39, 21 or 63, 27, 99, 33, people seldom commit to "multiples of 3". They extend "odd", a final digit or a range instead, and say yes to odd non-multiples of 3 (19, 29, 49, 23, 37) about as readily as to other multiples of 3.
- **prime_salient_hundred_chart**: Refinement of `hundred_chart_row_ranges` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment). The one change, taken from `prime_salient_concepts`, is that "prime numbers" is a familiar school-math category kept ready to hand, so it gets a fitted prior bonus over other compound rules. So after all-prime sets such as 2, 47 or 67, 31, 17, 61, people commit to "primes": they say yes to other primes more than to odd composite numbers, which the current best model gets backwards.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| irregular_threes_times_table ←selected | 0.00 | 0.00 | — (best) | 0.857 | yes |
| prime_over_odd_reading | 1.88 | 5.20 | no (within ~2·dse) | 0.131 | yes |
| repdigit_pattern_chart_concepts | 4.76 | 5.56 | no (within ~2·dse) | 0.007 | yes |
| prime_salient_marked_odd | 5.27 | 5.56 | no (within ~2·dse) | 0.004 | yes |
| hidden_divisibility_by_three | 7.23 | 3.17 | yes | 0.001 | yes |
| prime_salient_hundred_chart | 12.23 | 7.93 | no (within ~2·dse) | 0.000 | yes |
