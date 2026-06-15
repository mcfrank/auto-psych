# Theory Report — Experiment 3

## head_balance

**Hypothesis:** People judge a sequence as more random when its head proportion is
closer to 0.5, assessing perceived randomness solely by whether the sequence looks
like it came from a fair (unbiased) coin — they ignore alternation patterns and run
structure entirely.

**Motivation:** In the experiment 2 inner model loop, the inner loop independently
discovered an imbalance-only candidate (`iter0_candidate2`, hypothesis: "people only
care about H/T balance", posterior=0.095). This is the only single-cue hypothesis in
that experiment that captured non-trivial posterior mass and is not already present as
a seed model — all existing seed models either combine balance with another cue
(prototype_similarity, inner_loop_model) or ignore balance entirely (max_run_length,
rle_description_length, bayesian_diagnosticity uses it only indirectly). The four
top-ranked models from experiment 2 are statistically indistinguishable, suggesting
more diagnostic stimuli and a richer model set are needed to separate them.
Additionally, the ground truth `encoding_compressibility` model uses `imbalance` as
one of three weighted cues — isolating the balance mechanism tests whether that
component alone is sufficient to explain responses.

**Mechanism:** Score each sequence by its negative imbalance
(`score = -imbalance`), where `imbalance = |p_heads - 0.5|`. The sequence with
the smaller imbalance is predicted to be chosen as more random via
`p_left = sigmoid(β · (score_a − score_b) + bias)`. This is distinct from
`prototype_similarity` and `inner_loop_model`, which also penalize deviation from an
ideal alternation rate and cannot reduce to pure balance judgment. It is distinct from
`bayesian_diagnosticity`, which computes log Bayes factors against multiple structured
generators and incorporates run/alternation information. It is distinct from
`max_run_length` and `rle_description_length`, which use run-length features and
ignore balance entirely.
