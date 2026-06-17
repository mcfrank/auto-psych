# Theory Report — Experiment 2

## fewer_heads_proportion
**Hypothesis:** People judge the randomness of a sequence by its proportion of heads, preferring sequences with a lower proportion of heads as more random.
**Motivation:** The experiment 1 inner-loop report identified a model penalizing the absolute count of heads (`inner_loop_model`) as the best. However, a model based on the proportion of heads (`iter0_candidate2`) was second-best. Adding `fewer_heads_proportion` explicitly tests whether relative frequency (proportion) is a better predictor than absolute frequency.
**Mechanism:** This model assigns a score proportional to the negative of the sequence's proportion of heads (`p_a`, `p_b`), translating to a preference for sequences with a lower proportion of heads independent of length.

## short_streaks
**Hypothesis:** People judge sequences with shorter streaks (maximum run length of identical outcomes) as more random.
**Motivation:** While experiment 1 showed a strong preference for fewer heads, previous literature strongly suggests representativeness heuristic effects driven by streak avoidance. `iter0_candidate1` explored this but was lost in the iter1 manifest; adding it ensures this distinct baseline hypothesis is continually tested.
**Mechanism:** This model penalizes sequences directly based on their maximum run length (`max_run_a`, `max_run_b`), avoiding any combination with head proportions or other features.
