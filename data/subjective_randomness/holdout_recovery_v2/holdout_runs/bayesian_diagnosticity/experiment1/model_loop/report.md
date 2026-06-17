# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.638, elpd_loo=-412.07)
- Trials: 600
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.6384 | -412.07 |
| iter0_candidate0 | 0.1451 | -413.65 |
| iter1_candidate2 | 0.1134 | -414.15 |
| prototype_similarity | 0.0712 | -414.22 |
| iter0_candidate2 | 0.0130 | -416.32 |
| iter0_candidate1 | 0.0107 | -416.47 |
| iter1_candidate1 | 0.0074 | -416.79 |
| encoding_compressibility | 0.0009 | -418.15 |

## Hypotheses

- **iter1_candidate0**: People judge a sequence as more random-looking when its alternation rate is closer to an internal prototype ideal, and this proximity is computed quadratically: the randomness penalty grows with the *square* of the deviation from the ideal rate. Small departures from the prototype are disproportionately forgiven relative to large ones — matching a Gaussian similarity function around the prototype — rather than incurring a constant linear penalty for every additional unit of deviation as in an absolute-value model.
- **iter0_candidate0**: People judge a sequence as random-looking based entirely on how close its alternation rate is to an internal prototype ideal — the frequency at which a truly random sequence ought to switch between heads and tails. A sequence that alternates at the right rate looks most random; sequences that are either too streaky or too repetitively alternating look less random. The overall balance of heads versus tails in the sequence provides no additional information once alternation rate is accounted for.
- **iter1_candidate2**: People judge a sequence as more random-looking when its alternation rate is closer to 0.5 — the rate a fair coin produces. They have a fixed internal prototype at exactly 0.5, not a learned one; the only free parameter is how sensitively they respond to deviations from that prototype. A sequence whose switching rate is nearer to 0.5 is always preferred as more random, regardless of any other feature.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **iter0_candidate2**: People judge a sequence as more random when its proportion of heads and tails is closer to 50/50. Outcome balance is the single salient cue — a fair coin should produce roughly equal numbers of each outcome, so a sequence that deviates from that balance looks non-random. No other feature of the sequence, such as its switching rate or streak length, matters beyond this balance check.
- **iter0_candidate1**: People judge a sequence as more random when its longest streak of identical outcomes is shorter. A long run of repeated heads or tails is the single most cognitively salient violation of randomness; sequences where the maximum run is brief look random, and sequences containing even one long streak look non-random. No other feature of the sequence — overall head balance, alternation rate, or periodic structure — matters once maximum run length is accounted for.
- **iter1_candidate1**: People judge a sequence as random-looking based on how much periodic structure it contains. A truly random sequence should have no detectable, regularly-repeating pattern, so a sequence that cycles through outcomes in a predictable rhythm looks non-random. When comparing two sequences, people choose the one with lower periodicity as the more random-looking one.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 0.838 |
| iter0_candidate0 | 1.58 | 0.99 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 2.08 | 2.47 | no (within ~2·dse) | 0.160 |
| prototype_similarity | 2.14 | 0.98 | yes | 0.000 |
| iter0_candidate2 | 4.25 | 3.05 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 4.39 | 3.05 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 4.71 | 3.29 | no (within ~2·dse) | 0.001 |
| encoding_compressibility | 6.08 | 3.09 | no (within ~2·dse) | 0.000 |
