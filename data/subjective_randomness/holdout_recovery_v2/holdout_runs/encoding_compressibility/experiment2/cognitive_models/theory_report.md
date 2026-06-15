# Theory Report — Experiment 2

## max_run_length

**Hypothesis:** People judge a sequence as more random when its longest consecutive run
is shorter, because a long run (e.g., HHHHH) is the most compact single-symbol
run-length encoding unit and is the single most salient cue that a sequence is
non-random.

**Motivation:** In experiment 1, the inner model loop produced `iter1_candidate0`
(max-run hypothesis, posterior=0.324, ELPD=-221.29) but this model was never
promoted to the theory-agent's cognitive_models set. The ground truth for this
holdout run is `encoding_compressibility`, which uses `max_run_norm` as one of its
three penalty components. Including a pure `max_run_norm` model in the theory-agent
set gives the model loop a clean single-mechanism starting point that captures that
component. The Bayesian diagnosticity model (posterior=0.119) is the weakest of
the three competitive models, suggesting room for a more direct encoding-based
hypothesis.

**Mechanism:** Score each sequence as `−max_run_norm` (normalized longest run); the
sequence with the shorter longest run scores higher and is predicted more likely to
be chosen as random. This is distinct from `prototype_similarity` (which uses
imbalance and alternation proportion) and `bayesian_diagnosticity` (which computes a
Bayesian likelihood ratio vs. non-random generators) — it relies on a single
run-length encoding cue rather than aggregate statistics or generative models.

---

## rle_description_length

**Hypothesis:** People judge a sequence as more random when its run-length encoding
requires more blocks — the block count equals (alternations + 1), so more alternations
always means harder to compress and thus more random, with no ceiling or ideal rate.

**Motivation:** All three competitive models from experiment 1 are statistically
indistinguishable (all within ~2·dse of the best). `prototype_similarity` uses
alternation rate but penalizes BOTH too few AND too many alternations relative to a
learned ideal (non-monotonic). A pure encoding-compressibility account predicts a
strictly monotonic relationship: more alternations = more RLE blocks = harder to
compress = more random. These two functional forms make opposite predictions for
highly-alternating sequences like HTHTHTH (prototype_similarity rates them as
non-random because they deviate from the learned ideal; rle_description_length rates
them as maximally random). This distinction directly tests whether the alternation
effect is bounded (prototype) or unbounded (compressibility).

**Mechanism:** Compute `rle = (alts + 1) / n` (normalized RLE block count) for each
sequence; predict `p_left = sigmoid(β · (rle_a − rle_b) + bias)`. This is a
single-mechanism model that captures only the compression-length dimension of
encoding compressibility, using no imbalance or max-run signal.
