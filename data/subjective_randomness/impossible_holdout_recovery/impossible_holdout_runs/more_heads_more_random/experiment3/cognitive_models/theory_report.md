# Theory Report — Experiment 3

## power_law_heads
**Hypothesis:** People evaluate the randomness of a sequence based on the number of heads it contains, but their perception of randomness scales as an inferred power-law function of the head count.
**Motivation:** The `squared_heads_heuristic` (using heads squared) was the best-performing model in Experiment 2 (ELPD-LOO = -28.28), outperforming the absolute head-count and proportion-based models. A model from the inner loop (`iter1_candidate0`) using a parameterized power-law exponent was statistically distinguishable and very close in performance. Inferring the exponent directly allows us to determine the exact non-linear scaling of head counts on randomness perception.
**Mechanism:** Instead of fixing the exponent to 2 (as in `squared_heads_heuristic`), this model places a prior over the exponent and infers it from the data, transforming `h_a` to `h_a ** exponent` before logistic comparison.

## absolute_heads_lapse
**Hypothesis:** People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains, but their choices are subject to a constant lapse rate representing random guessing.
**Motivation:** The inner loop proposed a model (`iter1_candidate2`) that combined absolute head counts with a constant lapse rate, which performed reasonably well in the inner loop evaluation. It is plausible that participants occasionally guess or press the wrong button regardless of the stimuli.
**Mechanism:** The model calculates a logistic choice probability based strictly on the difference in absolute head counts (`h_a - h_b`), but mixes this probability with a 50% random guessing rate weighted by an inferred lapse parameter.
