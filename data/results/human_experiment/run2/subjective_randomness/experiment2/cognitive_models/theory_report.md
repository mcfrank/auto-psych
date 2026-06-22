# Theory Report — Experiment 2

## falk_konold_difficulty
**Hypothesis:** Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's (1997) Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.
**Motivation:** While the previous model loop found `encoding_compressibility` to be a poor fit compared to the messy prototype, this new hypothesis explores a different operationalization of compressibility/difficulty based strictly on Falk & Konold's established chunking metric (DP) rather than combining varied features like runs, periodicity, and imbalance.
**Mechanism:** Computes the sequence's difficulty predictor ($n_1 + 2 n_2$), normalized by the sequence length, and uses this score directly to predict randomness judgments: higher difficulty equals more random. It distinguishes itself by relying entirely on the structure of chunking (repetition vs alternation motifs) instead of evaluating continuous geometric or probability spaces.

## manhattan_messy_prototype
**Hypothesis:** Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.
**Motivation:** In the previous iteration, `inner_loop_model` (which used quadratic Euclidean distance) achieved the best ELPD-LOO posterior mass by a wide margin. This model proposes a refinement by hypothesizing that penalties accumulate linearly (absolute value) rather than quadratically, testing whether the "wide tolerance for near-ideal sequences" (a property of quadratic curves near zero) is truly what drives human responses or if a constant penalty slope fits better.
**Mechanism:** Implements exactly the same messy prototype and asymmetric alternation logic as `inner_loop_model`, but replaces the squared deviations with absolute deviations.
