# Theory Report — Experiment 3

## nonlinear_run_proportion
**Hypothesis:** People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes relative to the total sequence length, but their perception of this proportion is non-linear, following a power law.
**Motivation:** In the previous inner loop, `iter1_candidate0` achieved 28% posterior mass (indistinguishable from the best model), suggesting that while people rely heavily on the relative length of the longest streak, they may not evaluate it strictly linearly. Formalizing this as a distinct hypothesis allows us to test if diminishing or accelerating marginal returns better capture perception.
**Mechanism:** The model takes the normalized maximum run length (`max_run_norm_a`) and raises it to an inferred power `gamma` before applying the softmax comparison. This allows non-linear scaling of the proportion while remaining a distinct hypothesis from the purely linear `inner_loop_model`.

## surprising_run_length
**Hypothesis:** People judge the randomness of a sequence by assessing how much the absolute length of its longest streak of identical outcomes exceeds the natural logarithmic growth expected for a sequence of that total length.
**Motivation:** The previous report showed absolute streak length (`iter1_candidate1`) failed severely, while normalized streak length (`inner_loop_model`) succeeded. However, true maximum streak length scales logarithmically with sequence length, not linearly. This model tests whether people intuitively evaluate streaks against a logarithmically growing baseline expectation rather than a simple linear proportion.
**Mechanism:** Instead of dividing by `N`, the model calculates an expected baseline run length using `log2(N)`. The subjective score is the difference between the absolute maximum run (`max_run_a`) and this logarithmic expectation.
