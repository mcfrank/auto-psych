# Refinement menu

The models you may refine, other than the incumbent `erlang_width_compressed_intervals`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### parity_restricted_range_conjunctions — rank 1, 94.6 ± 13.7 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1990.3

**Hypothesis:** People can build compound concepts that join a numerical range with the examples' parity ("even numbers between 20 and 60"), but considering such a two-part concept carries a cognitive cost, so these conjunctions get only a fitted, reduced share of prior belief alongside the simple rules and plain ranges of exception-tolerant Bayesian concept learning on a compressed number line. Once considered, the size principle strongly favours these small compound concepts, so people fill in the range around the examples but reject immediate neighbours of the wrong parity (and odd-one-out neighbours of an outlier example) far more than a learner who only has plain ranges.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/parity_restricted_range_conjunctions.py`

### log_compressed_number_line_intervals — rank 2, 95.2 ± 14.3 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1991.0

**Hypothesis:** People represent numerical-magnitude concepts ("numbers between 40 and 60") as regions on a logarithmically compressed mental number line, not on the linear 1-100 scale: when they infer which stretch of numbers the program produces, a stretch's size, and how likely the examples are to fall where they do within it, are judged in compressed (Weber-like) units, while arithmetic and digit rules are judged as before. Because the same numerical step is subjectively smaller among large numbers, people extend a magnitude region upward beyond the largest example more readily than downward below the smallest, and generalise across the span between the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/log_compressed_number_line_intervals.py`

### parity_restricted_compressed_intervals — rank 3, 96.1 ± 13.6 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1991.9

**Hypothesis:** Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all ranges of 1-100, with range members pictured as sampled along a compressed mental number line, but their vocabulary of magnitude concepts also contains parity-restricted ranges ("even numbers between 20 and 40", "odd numbers between 51 and 67"), which share a fitted portion of the prior given to ranges. The one change is these conjunctive parity-and-magnitude concepts: when the examples share a parity, people generalise to nearby numbers of the same parity but reject immediate neighbours (which flip parity) and near numbers around an off-pattern example more than a learner with only plain ranges would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/parity_restricted_compressed_intervals.py`

### round_boundary_interval_prior — rank 4, 112.1 ± 14.2 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2007.8

**Hypothesis:** People infer numerical-range concepts anchored on round numbers: when they consider that the program produces "a range of numbers", they think of ranges whose edges fall on decade boundaries ("from 20 to 50", "the 80s", "up to 100") as far more plausible than ranges with arbitrary edges such as 27-92 (the strength of this round-boundary preference fitted), while otherwise doing the same exception-tolerant, tempered Bayesian learning over arithmetic rules and ranges sampled along a compressed number line. Because the preferred range is snapped outward to the nearest decade edges, people fill in the span between the examples, generalise within the examples' decades but not across a decade boundary just beside an example, and extend above an example ending in a low digit more than below it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/round_boundary_interval_prior.py`

### log_compressed_interval_sampling — rank 5, 112.1 ± 14.3 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2007.9

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they picture the examples of an interval concept as sampled along a compressed (roughly logarithmic) mental number line, so within a range larger numbers are individually less likely to be drawn and the size principle charges less for extending a range upward than downward. The one change is this compressed-number-line sampling of interval members (degree of compression fitted; no compression gives back essentially the original model), which predicts that people generalise past the largest example more than below the smallest, closing the critique that humans extend upward more than the symmetric models predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/log_compressed_interval_sampling.py`

### graded_profile_with_neighborhoods — rank 6, 157.5 ± 26.3 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2053.2

**Hypothesis:** Refinement of `graded_feature_profile_matching`: people still judge membership by matching the target against a frequency-weighted profile of the examples' properties (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but the profile also contains each example's own numerical neighbourhood ("numbers close to 27") as one more graded property, with a fitted width, weighted by the same distinctiveness rule and shared in proportion to how many examples fall near it. The one change is this soft neighbourhood property: targets just one to three away from an example gain strong support that decays steeply within a few units, closing the critique that human generalisation falls off over short numerical distances far more sharply than the current models predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/graded_profile_with_neighborhoods.py`

### weber_scaled_profile_neighborhoods — rank 7, 244.0 ± 30.5 nats behind the best (8.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2139.8

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people still judge membership by matching the target against a frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties plus each example's own soft numerical neighbourhood, but that neighbourhood lives on a logarithmically compressed (Weber-like) mental number line rather than the linear 1-100 scale. The one change is this compressed neighbourhood (grafted from the compressed-number-line models): "close to 8" covers only a unit or two while "close to 80" covers many, the neighbourhood reaches further above an example than below it, and a broader neighbourhood counts as a less distinctive property — so people extend upward beyond the examples more than downward, as the critique of the incumbent reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/weber_scaled_profile_neighborhoods.py`

### rule_vs_range_size_principle — no comparison row

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant Bayesian concept learning over the same 45 arithmetic and digit rules and width-weighted ranges on a compressed number line, but they do not apply a single size principle to both kinds of concept. When they weigh how well a symbolic rule ("ends in 2", "even", "multiples of 4") explains the examples, their sensitivity to its size differs, by a fitted factor, from their sensitivity when they weigh a numerical range. The one change is this kind-specific strength of the size principle, which decides whether more examples mainly tighten the range or mainly sharpen the choice between narrow and broad rules (such as "ends in 2" against "even").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/rule_vs_range_size_principle.py`

### luce_concept_selection — no comparison row

**Hypothesis:** People do not average their predictions over every concept they find plausible; they commit to the concept that best explains the examples, choosing among the candidate rules and numerical ranges (same exception-tolerant learner over the 45 rules and compressed-line ranges with a moderate-width prior) by a Luce choice whose sharpness on the posterior is fitted, from soft averaging to near-single-hypothesis selection. Because the winning concept takes nearly all the weight, when every example shares a final digit or a parity people extend that rule almost unanimously and reject parity-violating neighbours more firmly than a hedging Bayesian learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/luce_concept_selection.py`

### first_example_anchored_concept — no comparison row

**Hypothesis:** People anchor their concept on the first example they see: they take the first example as more surely produced by the program than the later ones (by a fitted factor), treating mainly the later examples as possible slips, so a later example that breaks the pattern set up by the first is discounted as an exception while the first example itself is rarely explained away (otherwise they do the same exception-tolerant, tempered Bayesian concept learning over arithmetic rules and moderate-width ranges on a compressed number line). The same examples shown in a different order therefore lead to different generalisations: people generalise around the first example and rules containing it, and reject neighbours of an off-pattern example more when it comes later in the list.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/first_example_anchored_concept.py`

### final_digit_salient_rules — no comparison row

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and moderate-width ranges sampled along a compressed number line, but they do not consider all rules equally plausible beforehand — rules that can be read straight off a numeral's final digit (odd, even, "ends in d") are perceptually salient and get a fitted prior boost over the other arithmetic and digit rules. The one change is this final-digit salience in the rule prior: when all examples share a final digit or a parity, people commit to that rule more firmly than to a range, so they endorse distant same-ending and same-parity numbers more and reject opposite-parity neighbours (one step from an example) more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_salient_rules.py`

### family_salience_rule_prior — no comparison row

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all ranges of 1-100 (members sampled along a compressed number line, Erlang-shaped width prior), but they do not find every kind of rule equally plausible beforehand — whole families of rules differ in salience (parity, final-digit, contained-digit, multiples, powers, and special sets such as primes or squares), with the share of prior belief each family receives fitted and split equally among its members. The one change is this family-salience prior over rules, replacing the flat one: if parity and final-digit rules are especially salient, people extend a shared final digit to distant targets almost unanimously and reject opposite-parity neighbours of same-parity examples more firmly than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/family_salience_rule_prior.py`

## Pruned models (out of the set; narrowest margin first)

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 10.1 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 11

**Hypothesis:** People do tempered Bayesian concept learning over arithmetic rules and numerical intervals, but they do not weigh the examples equally: how much each example counts as evidence about the concept depends on its position in the order shown, rising or falling smoothly from the first example to the last (primacy or recency, the direction and strength fitted). So the same examples shown in a different order lead to different generalisations — an example that breaks the others' pattern is discounted less when it comes in the favoured position, and people generalise around it more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_evidence.py`

### exception_tolerant_tempered_mixture — pruned (experiment1 end of experiment)

**Margin:** 11.9 nats behind parity_restricted_range_conjunctions (2.2× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they believe the program often slips and produces a number outside its concept, so they tolerate exceptions — the fraction of examples assumed to be off-concept noise is a fitted parameter instead of a fixed 10%. The one change is this fitted exception rate: when most but not all examples share a regularity (e.g. three of four end in 2), the rule that covers the majority survives the exception, so people extend a partly shared last digit or arithmetic property to distant targets, and lean less on tight numerical intervals, than a learner who treats every example as strictly from the concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exception_tolerant_tempered_mixture.py`

### decisive_exception_tolerant_mixture — pruned (experiment1 end of experiment)

**Margin:** 12.0 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 11

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not report their belief that the target is in the concept as a graded probability — they turn it into a near-categorical decision, so the odds of saying yes are the posterior odds raised to a fitted decisiveness exponent. The one change is this response-decisiveness transform: targets the learner thinks are probably in (or out of) the concept are endorsed (or rejected) more consistently than the posterior itself implies, making answers to each item more unanimous, sets more sharply split between narrow and broad generalisation, and near-but-off-rule neighbours of an outlier example rejected more firmly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decisive_exception_tolerant_mixture.py`

### upward_counting_progressions — pruned (experiment1 end of experiment)

**Margin:** 13.0 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules and numerical intervals, but their concept vocabulary also contains counting-up sequences: arithmetic progressions that start at some number and climb by a fixed step to 100 ("16, 24, 32, … upward"), this whole family sharing a fitted share of the prior. Because such a sequence must start at or below the smallest example and continues indefinitely upward, people fill in the gaps between the examples at the inferred spacing and extend their generalisation above the examples more readily than below them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/upward_counting_progressions.py`

### graded_feature_profile_matching — pruned (experiment1 end of experiment)

**Margin:** 13.4 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 11

**Hypothesis:** Refinement of `common_feature_profile_matching`: people still judge membership by matching the target against the arithmetic and surface properties of the examples (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but a property enters the profile in proportion to how many of the examples have it rather than only when every example has it. The one change is this graded, frequency-weighted profile (its steepness fitted by one exponent), so a target that shares a final digit or another property with only some of the examples still gains support — closing the critique that people endorse partial last-digit matches far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/graded_feature_profile_matching.py`

### decision_threshold_exception_tempered — pruned (experiment1 end of experiment)

**Margin:** 17.7 nats behind parity_restricted_range_conjunctions (2.6× dse)

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not answer yes with probability equal to their belief that the target is in the concept; instead they compare that belief against an internal criterion and respond near-deterministically, with a fitted decision sharpness and criterion. The one change is this thresholded (sharpened) decision rule on the posterior: beliefs above the criterion are pushed toward a confident yes and beliefs below it toward a confident no, so responses to each item are more unanimous, sets differ more in breadth, and endorsement falls off more steeply with numerical distance than the graded posterior itself implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decision_threshold_exception_tempered.py`

### place_value_grid_regions — pruned (experiment1 end of experiment)

**Margin:** 23.6 nats behind parity_restricted_range_conjunctions (2.9× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules, but they represent magnitude on a decimal place-value grid (tens digit by ones digit) rather than on a continuous number line: their non-arithmetic concepts are rectangular regions of that grid — a range of tens digits crossed with a range of final digits, such as "the 40s to 90s ending in 2" or "the 20s and 30s" — weighed by the size principle. This disagrees most sharply with the current best (number-line interval) model on targets one step from an example across a decade boundary or a final-digit change (e.g. 26 or 28 for an outlier 27 among 92, 82, 42), which it predicts people reject more, and on distant targets sharing a tens or final digit with the examples, and filling the grid block the examples span, which it predicts people endorse more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/place_value_grid_regions.py`

### graded_profile_with_example_span — pruned (experiment1 end of experiment)

**Margin:** 33.2 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people still match the target against a frequency-weighted, distinctiveness-weighted profile of the examples' properties plus each example's soft numerical neighbourhood, but the profile also contains the numerical span the examples cover ("numbers between the smallest and largest example") as one more property, shared by every example, counting more the narrower the span, with soft edges that fade over the same fitted neighbourhood width. The one change is this span (interval) property: targets lying between the examples but far from any single one gain support, closing the critique that people generalise over the range spanned by the examples more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/graded_profile_with_example_span.py`

### graded_profile_with_spanned_interval — pruned (experiment1 end of experiment)

**Margin:** 35.3 nats behind parity_restricted_range_conjunctions (2.8× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people match the target against the same frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties and each example's own soft numerical neighbourhood, but when there are two or more examples the profile also contains the numerical interval they span ("numbers between 23 and 41") as one more property, shared by every example, weighted by the same rarity rule (narrow spans count more), and with soft edges of the same fitted width. The one change is this spanned-interval property: targets lying between the examples gain support even when far from every individual example, closing the critique that people generalise over the examples' range more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/graded_profile_with_spanned_interval.py`

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Margin:** 45.5 nats behind parity_restricted_range_conjunctions (4.8× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — the evidence from the examples is tempered by a single fitted exponent, so a few examples that happen to sit close together do not overwhelm a broad rule such as "even" or "ends in 2" the way strict strong sampling would. The one change is this evidence-weighting exponent (sharpness of the size principle), which shifts belief toward broad rules when examples share a regularity and lets generalisation tighten less with more examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_mixture.py`

### memory_limited_example_recall — pruned (experiment1 end of experiment)

**Margin:** 48.5 nats behind parity_restricted_range_conjunctions (4.9× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numerical intervals, but they reason from an imperfect memory of the examples: each example is independently retained with some probability, and they generalise from only the examples they hold in mind. Because a judgement is often based on one or two of the examples rather than all of them, a target that shares a property (such as a final digit or a divisor) with just one example is often endorsed, and generalisation tightens less with more examples than an ideal learner who uses every example would predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/memory_limited_example_recall.py`

### tempered_size_principle_digit_unions — pruned (experiment1 end of experiment)

**Margin:** 49.9 nats behind parity_restricted_range_conjunctions (4.9× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but their rule vocabulary also includes disjunctive last-digit concepts such as "ends in 2 or 7" or "ends in 1, 6 or 0" (unions of two to four final digits), each made less plausible a priori by a fitted penalty per extra digit. The one change is this enlarged hypothesis space: when the examples' final digits differ, people can still infer a final-digit rule covering every example, so they say yes to distant targets that share a final digit with any one example far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_digit_unions.py`

### soft_threshold_posterior_decision — pruned (experiment1 end of experiment)

**Margin:** 50.8 nats behind parity_restricted_range_conjunctions (4.2× dse)

**Hypothesis:** People infer the hidden concept by the same simple Bayesian concept learning over arithmetic rules and numerical intervals as the basic mixture model, but they do not answer "yes" with a probability equal to their belief: they compare their belief that the target is in the concept against an internal criterion and answer yes when it clears that criterion, with some noise around it. This soft-threshold decision rule makes answers more all-or-none than the graded posterior (near-unanimous yes for well-supported targets, near-unanimous no for weakly supported ones), with a fitted criterion setting an overall bias toward yes or no.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/soft_threshold_posterior_decision.py`

### common_feature_profile_matching — pruned (experiment1 end of experiment)

**Margin:** 53.5 nats behind parity_restricted_range_conjunctions (3.1× dse)

**Hypothesis:** People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/common_feature_profile_matching.py`

### tempered_size_principle_rules_intervals — pruned (experiment1 end of experiment)

**Margin:** 77.8 nats behind parity_restricted_range_conjunctions (4.2× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but they weigh the evidence of the examples less sharply than an ideal learner, so the size principle is tempered (the likelihood of the examples is raised to a fitted power below or above one). The one change is this fitted evidence-strength exponent: a weak size principle lets broad rules such as "even" or "ends in 2" keep their prior weight against tight intervals, so people extend rules to distant targets and tighten generalisation with more examples less than the ideal learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_rules_intervals.py`

### size_weighted_prior_mixture — pruned (experiment1 end of experiment)

**Margin:** 79.9 nats behind parity_restricted_range_conjunctions (3.2× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning (same 45 rules, all intervals of 1-100, size-principle likelihood), but their prior is not flat within each kind of concept — it favours broader concepts in proportion to a fitted power of each concept's size. The one change is this size-weighted prior: broad rules such as "even" or "ends in 3" start out more plausible than narrow ones like "multiples of 4" or tight intervals, a fixed head start that (unlike a tempered size principle) does not grow or shrink with the number of examples, so people extend a shared regularity to distant numbers more often and endorse near numbers that break it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_weighted_prior_mixture.py`

### example_subset_memory_bayes — pruned (experiment1 end of experiment)

**Margin:** 92.7 nats behind parity_restricted_range_conjunctions (6.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same arithmetic rules and numerical intervals as an ideal learner (strong sampling, size principle), but under a working-memory limit: each example shown enters their inference only with some fitted probability, so they often generalise from just a subset of the examples. Reasoning from one or two remembered examples lets a rule shared by only some examples (e.g. "ends in 3" from 23 in {16, 23}) win, and weakens the size-principle tightening that more examples would normally produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/example_subset_memory_bayes.py`

### set_adaptive_attention_exemplar — pruned (experiment1 end of experiment)

**Margin:** 132.2 nats behind parity_restricted_range_conjunctions (4.5× dse)

**Hypothesis:** People judge membership by summed similarity to the stored examples (an exemplar model), where the similarity metric adapts to the set: attention goes to the dimensions on which the examples agree — their final digit, parity, and remainders modulo 3, 4 and 5 — and to numerical magnitude in proportion to how tightly the examples cluster, with magnitude compared on a logarithmic (Weber-like) scale. A target is endorsed in proportion to its total similarity to the examples (Luce choice against a fixed criterion), so targets between several examples collect similarity from all of them, generalisation extends further upward than downward, and a final digit shared by every example becomes a near-mandatory match for distant targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/set_adaptive_attention_exemplar.py`

### exemplar_digit_feature_similarity — pruned (experiment1 end of experiment)

**Margin:** 136.6 nats behind parity_restricted_range_conjunctions (4.4× dse)

**Hypothesis:** People judge membership by surface-feature similarity to the individual examples, not by inducing a rule shared by the whole set: they compare the target with each example on a few salient digit and arithmetic features (its last digit, its tens digit, its parity, and its remainder when divided by three), and the more features the target shares with a given example the more similar it feels, with similarity falling off exponentially with the number of mismatching features. The yes probability grows with the target's average similarity to the examples, so a target that shares only its final digit with one example (in a set without a common final digit) still earns credit, and plain numerical distance plays no role beyond sharing a tens digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_digit_feature_similarity.py`

### common_feature_matching — pruned (experiment1 end of experiment)

**Margin:** 173.6 nats behind parity_restricted_range_conjunctions (5.8× dse)

**Hypothesis:** People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/common_feature_matching.py`

### tempered_size_principle — pruned (experiment1 end of experiment)

**Margin:** 188.7 nats behind parity_restricted_range_conjunctions (5.0× dse)

**Hypothesis:** People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle.py`

### numeral_edit_distance_similarity — pruned (experiment1 end of experiment)

**Margin:** 227.6 nats behind parity_restricted_range_conjunctions (6.3× dse)

**Hypothesis:** People judge membership by how similar the target looks, as a written numeral, to each example: similarity falls off exponentially with the number of digit edits (substitutions, insertions or deletions of a digit) needed to turn the target's numeral into an example's, and the yes probability grows with the target's average similarity across the examples. Arithmetic rules and number-line magnitude play no separate role, so a target sharing its tens digit or its final digit with many of the examples (e.g. 62 for 92, 82, 42) is endorsed even when it is far away, while one sharing no digit position with any example is rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/numeral_edit_distance_similarity.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 229.9 nats behind parity_restricted_range_conjunctions (7.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 277.6 nats behind parity_restricted_range_conjunctions (5.9× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
