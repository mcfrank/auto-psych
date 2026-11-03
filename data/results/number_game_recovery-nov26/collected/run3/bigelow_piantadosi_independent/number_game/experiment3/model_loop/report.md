# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **square_pair_concepts** (posterior=1.000, elpd_loo=-3469.92)
- Trials: 6750
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| square_pair_concepts | 1.0000 | -3469.92 |
| forward_continuation_spacing_sparsity | 0.0000 | -3482.62 |
| forward_continuation_response_bias | 0.0000 | -3483.02 |
| widest_gap_sparsity_round_veto | 0.0000 | -3483.99 |
| round_landmark_veto | 0.0000 | -3485.44 |
| first_example_anchored_trust | 0.0000 | -3485.97 |
| round_veto_spacing_sparsity | 0.0000 | -3484.89 |
| pedagogical_example_selection | 0.0000 | -3487.97 |

## Hypotheses

- **square_pair_concepts**: People can read a set of examples as "a number and its square" (for example 8 and 64, 9 and 81, or 4 and 16), so the pair made up of a number and its square is one more concept they consider for the program. It is priced like the other two-part rules, plus its own fitted prior cost or bonus, and is added to the current best model's concept search, which stays the same in every other way. Because the concept has only two members, the size principle makes it win whenever the examples are exactly such a pair. So after 8, 64 people say no to almost every other number, including other even numbers, multiples of 8 and powers of 2 such as 24, 32, 56 or 72, which the current best model predicts they accept.
- **forward_continuation_spacing_sparsity**: I refined the incumbent `forward_continuation_response_bias` and kept all of it: people infer the program with the set-size-graded privileged-fives concept learner, veto round targets that no round example supports, and at the answer stage lean toward yes above the largest example and toward no below the smallest. The one change is taken from `example_spacing_sparsity_bias`: people also read the average gap between neighbouring examples as a clue to how sparse the program's output is, so the wider the examples are spread, the less ready they are to say yes to any target, on top of the upward lean. So after widely spaced sets such as 8, 64 or 14, 98, 91 people say yes less often than the incumbent predicts, even above the examples, and after tightly clustered sets they say yes more often.
- **forward_continuation_response_bias**: At the answer stage people treat the examples as what the program has produced so far, and they assume it runs upward. So, beyond what the concept they inferred implies, they lean toward yes for a target above the largest example and toward no for a target below the smallest. The strength of this lean is fitted, and targets inside the examples' span are unaffected. Concept inference is the current best model's, unchanged (including its round-number veto). So after 32, 4, 16, a target such as 40 or 60 gets more yeses than 2 or 3, even where the inferred concept rates them equally.
- **widest_gap_sparsity_round_veto**: I refined `round_veto_spacing_sparsity` and kept all of it: the set-size-graded privileged-fives concept learner, the veto of round targets (10, 20, ..., 100) on a fitted share of trials when no example is round, and a fitted lowering of yes answers the sparser the program's output looks. The one change is how people judge that sparsity: not from the average gap between neighbouring examples but from the widest empty stretch between them, because a long run of numbers the program skipped is what signals that it produces few numbers. So a set with one big hole, such as 27, 42, 82, 92, makes people as reluctant to say yes as an evenly spread set with the same hole, even though its average gap is much smaller.
- **round_landmark_veto**: Round numbers (10, 20, ..., 100) are landmarks on the mental number line rather than ordinary outputs, so people check a round target against the examples before accepting it: if none of the examples is itself a round number, on a fitted share of such trials they doubt that the program produces round numbers and answer no, whatever concept they inferred. Concept inference itself is unchanged from the current best model, so the claim shows up only after sets with no multiple of 10 (e.g. 8, 64 or 32, 4, 16), where people say no to 10, 20, 30 or 70 more often than the concept they inferred (e.g. "even numbers") implies.
- **first_example_anchored_trust**: People build their guess about the program from the first example they see and then check each later example against that guess, so the first example anchors the concept: it is less readily set aside as a stray number than any later example, at a fitted rate. Concept inference is otherwise that of the current best model, with the round-number veto kept. So a rule or range that fits the first example but misses a later one survives better than one that fits the later examples but misses the first, and a target that shares the first example's rule or region gets more yeses than one that shares only a later example's.
- **round_veto_spacing_sparsity**: I refined the incumbent `round_landmark_veto` and kept all of it: people infer the program with the set-size-graded privileged-fives concept learner, and when no example is a round number they veto round targets (10, 20, ..., 100) on a fitted share of trials. The one change is taken from `example_spacing_sparsity_bias`: people also read how far apart the examples are as a clue to how sparse the program's output is, so the wider the average gap between neighbouring examples, the less ready they are to say yes to any target, beyond what the inferred concept implies. So after widely spaced sets such as 3, 48, 99 or 8, 64, people say yes less often than the incumbent predicts even to targets that fit the inferred rule or range, and after tightly clustered sets they say yes more often.
- **pedagogical_example_selection**: People do not treat the examples as random draws from the program's output. They assume a helpful teacher picked them to be diagnostic, favouring telling numbers that few candidate stretches of the number line contain, namely those near the ends of 1-100. So a concept that contains many such telling numbers seems unlikely when none of them was shown ("if the program reached the 90s, I would have been shown one"). The concept's effective size is weighted by how diagnostic its members are, at a fitted strength, in place of the plain size principle. Everything else in the current best model is kept, including its round-number veto and upward response lean.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| square_pair_concepts ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| forward_continuation_spacing_sparsity | 12.70 | 7.26 | no (within ~2·dse) | 0.000 | yes |
| forward_continuation_response_bias | 13.10 | 7.71 | no (within ~2·dse) | 0.000 | yes |
| widest_gap_sparsity_round_veto | 14.07 | 8.52 | no (within ~2·dse) | 0.000 | yes |
| round_veto_spacing_sparsity | 14.97 | 7.70 | no (within ~2·dse) | 0.000 | yes |
| round_landmark_veto | 15.52 | 8.10 | no (within ~2·dse) | 0.000 | yes |
| first_example_anchored_trust | 16.05 | 8.12 | no (within ~2·dse) | 0.000 | yes |
| pedagogical_example_selection | 18.06 | 9.00 | yes | 0.000 | yes |
