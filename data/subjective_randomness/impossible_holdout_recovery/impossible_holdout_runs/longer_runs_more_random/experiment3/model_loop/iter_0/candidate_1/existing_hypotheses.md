# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1680.51

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1895.51

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -861.01

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## ideal_run_proportion  — posterior 0.114, ELPD-LOO -783.15

People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.

## absolute_ideal_run  — posterior 0.000, ELPD-LOO -50508.74

People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length.

## pure_periodicity_penalty  — posterior 0.000, ELPD-LOO -1872.48

People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random.

## inner_loop_model  — posterior 0.477, ELPD-LOO -781.78

Best PyMC model found by the inner model-improvement loop.

## nonlinear_run_proportion  — posterior 0.409, ELPD-LOO -781.83

People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes relative to the total sequence length, but their perception of this proportion is non-linear, following a power law.

## surprising_run_length  — posterior 0.000, ELPD-LOO -837.64

People judge the randomness of a sequence by assessing how much the absolute length of its longest streak of identical outcomes exceeds the natural logarithmic growth expected for a sequence of that total length.

## iter0_candidate0

People judge the randomness of a sequence by comparing its maximum run proportion to a subjective ideal proportion, but they penalize deviations from this ideal using a squared distance, causing extreme deviations to seem disproportionately less random than minor ones.
