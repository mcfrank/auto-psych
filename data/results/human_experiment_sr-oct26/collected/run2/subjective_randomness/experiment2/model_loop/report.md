# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **tally_span_switch_ideal_periodic_unit** (posterior=1.000, elpd_loo=-2220.89)
- Trials: 5120
- Models compared: 5

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| tally_span_switch_ideal_periodic_unit | 1.0000 | -2220.89 |
| tally_span_switch_ideal_triplet_variety | 0.0000 | -2234.48 |
| tally_span_switch_rate_ideal_2 | 0.0000 | -2242.71 |
| pair_normalized_tally_switch_contrast | 0.0000 | -2243.90 |
| tally_span_switch_ideal_periodic_unit_2 | 0.0000 | -2242.21 |

## Hypotheses

- **tally_span_switch_ideal_periodic_unit**: Refinement of the incumbent `tally_span_switch_rate_ideal_2`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, a personal left/right lean), and still judge switching by closeness to a shared ideal switching rate with a person-specific weight — but they also spot a short unit being repeated (strict alternation HTHT..., or motifs like HHTHHT and HTTTHTTT, any unit of two or more flips repeated at least twice through the whole sequence), and a sequence with such a visible repeating pattern looks designed rather than random. The one change is this shared penalty on visible periodicity (taken from the alternation-ideal models), which lets the switching ideal sit higher so that heavy switching is not over-penalised, addressing the critiques (surviving FDR) that people choose periodic period-3/4 motifs less often, and the more-switching sequence among high-switch pairs more often, than the incumbent predicts.
- **tally_span_switch_ideal_triplet_variety**: Refinement of the incumbent `tally_span_switch_rate_ideal_2`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, personal left/right lean), and still judge switching by closeness to an ideal switching rate with a person-specific weight — but they also register the variety of short local patterns, so a sequence whose overlapping three-flip chunks are mostly different from one another looks more random than one that recycles the same few chunks (strict alternation HTHTHTHT uses only HTH and THT; repeating motifs such as HHTHHT use only three). The one change is a shared reward on the share of distinct three-flip chunks (taken from `tally_span_switch_triplet_variety`), which lets the switching ideal sit higher while rejecting regular high-switch sequences, addressing the critiques (surviving FDR) that people choose periodic-motif sequences less, the more-switching sequence among high-switch pairs more, and the sequence with more distinct triplets at equal switch counts more than the incumbent predicts.
- **tally_span_switch_rate_ideal_2**: Refinement of the incumbent `iter2_candidate3`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (with person-specific, length-scaled sensitivity and a personal left/right lean), and still attend to how often the coin switches sides with person-specific strength — but switching is no longer rewarded "the more the better": people expect a random coin to switch at a particular rate (somewhat above one half), so a sequence that switches less than that looks streaky and one that switches more, up to strict alternation like HTHTHTHT, looks too regular. The one change replaces the incumbent's linear switching reward with a closeness-to-an-ideal-switching-rate judgement (a shared ideal rate, person-specific weight), addressing the critiques (surviving FDR) that people pick a perfect alternator far less often than the linear reward implies and that the alternation preference changes shape among high-switching pairs.
- **pair_normalized_tally_switch_contrast**: People judge the two sequences against each other, not one at a time: each person still compares each sequence's running heads-minus-tails tally span with their own expected span and its switching rate with an ideal switching rate (with a personal left/right lean), but the difference between the two sequences on each cue is weighed relative to how far the pair as a whole sits from the ideal (divisive contrast normalisation). A given gap decides the choice sharply when both sequences are near the ideal and only weakly when both are far from it, so the same sequence is chosen more or less decisively depending on its partner — for instance, between two heavily switching sequences people barely penalise the one that switches more, unlike when it is paired with a moderately switching sequence.
- **tally_span_switch_ideal_periodic_unit_2**: Refinement of `tally_span_switch_rate_ideal` (running heads-minus-tails tally judged by closeness of its span to a personal expected span, switching judged by closeness to a shared ideal rate with person-specific weight, personal left/right lean): people additionally notice when a sequence is visibly built by repeating a short unit — strict alternation HTHTHTHT, HHTHHT, HHTTHHTT, HTTTHTTT — and see such a repeating pattern as designed, not random, regardless of its span or switching rate. The one change is a shared penalty on sequences that are exactly periodic (a unit repeated at least twice), which lets the switching ideal stop doing the work of rejecting strict alternation, addressing the critiques (surviving FDR) that people choose periodic-motif sequences less often, and the more-switching sequence among high-switch pairs more often, than the current models predict.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| tally_span_switch_ideal_periodic_unit ←selected | 0.00 | 0.00 | — (best) | 0.880 | yes |
| tally_span_switch_ideal_triplet_variety | 13.59 | 5.76 | yes | 0.094 | yes |
| tally_span_switch_ideal_periodic_unit_2 | 21.32 | 7.08 | yes | 0.026 | yes |
| tally_span_switch_rate_ideal_2 | 21.82 | 6.67 | yes | 0.000 | yes |
| pair_normalized_tally_switch_contrast | 23.01 | 6.83 | yes | 0.000 | yes |
