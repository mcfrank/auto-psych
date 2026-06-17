# Theory Report — Experiment 2

## length_scaled_head_difference
**Hypothesis:** People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger.
**Motivation:** The `inner_loop_model` relying strictly on absolute head count dominated the previous iteration. We refine this by proposing that a difference of e.g. 2 heads is far more noticeable when comparing sequences of lengths 3 vs 4 than lengths 7 vs 8, so we scale the linear difference by the sum of sequence lengths.
**Mechanism:** The model computes the difference in absolute head counts `h_a - h_b` and divides it by `n_a + n_b` before applying the softmax scaling factor `tau`. This isolates length-dependent sensitivity from the absolute linear difference hypothesis.

## squared_heads_heuristic
**Hypothesis:** People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts.
**Motivation:** Previous hypotheses tested logarithmic growth and proportional models and performed poorly relative to the linear absolute model. This model proposes a super-linear (squared) relationship where extreme head counts exhibit an outsized effect compared to moderate ones.
**Mechanism:** The model calculates `(h_a)^2 - (h_b)^2` as the decision variable. This is a distinct mechanism from the linear difference model because a fixed difference of 1 head has a larger impact on the choice probability when the total number of heads is large.
