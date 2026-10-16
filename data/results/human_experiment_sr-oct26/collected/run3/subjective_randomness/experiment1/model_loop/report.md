# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **length_normalised_chance_vs_motif** (posterior=0.999, elpd_loo=-995.13)
- Trials: 2560
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| length_normalised_chance_vs_motif | 0.9992 | -995.13 |
| bayesian_chance_vs_repeating_motif_2 | 0.0004 | -1002.98 |
| mismatch_noise_repeating_motif | 0.0002 | -1003.75 |
| bayesian_chance_vs_motif_beta_trick_coin | 0.0002 | -1003.88 |
| hot_hand_only_suspicion | 0.0000 | -1005.17 |
| mismatch_noise_paired_comparison | 0.0000 | -1021.59 |
| bayesian_chance_vs_repeating_motif | 0.0000 | -1016.27 |
| gist_tail_probability_test | 0.0000 | -1016.55 |

## Hypotheses

- **length_normalised_chance_vs_motif**: Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is in how that evidence is weighed: people judge the evidence per flip rather than in total, so the log evidence difference is divided by a fitted power of the sequence length — the same per-flip regularity is judged about as decisively in a short pair as in a long one, instead of long sequences automatically yielding near-certain choices — addressing the critique that people are more decisive on short pairs (relative to long) than the incumbent predicts and that it over-penalises long perfect alternation.
- **bayesian_chance_vs_repeating_motif_2**: Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent a fair coin (which they believe over-alternates) explains it better than a "regular" generator, with the regular generators' unknowns averaged out normatively. The single change is one more regular generator in their hypothesis space: a "repeating motif" process that writes a short pattern (one to four flips long, e.g. H, HT, HHT, HHTT) and keeps copying it with an occasional slip (a fitted slip rate), so sequences that are near-repetitions of a short motif — perfect alternation and period-3/4 patterns above all — are explained as regular and look less random, addressing the critique that the incumbent under-penalises perfect alternation and periodic motifs.
- **mismatch_noise_repeating_motif**: Refinement of `mismatch_noise_paired_comparison`: people still compare the two sequences flip against flip, so matching positions cancel and the noise in the comparison grows with the number of positions at which the two sequences differ (making short and near-identical pairs judged more decisively), and each sequence's evidence is still the Bayesian "fair coin believed to over-alternate versus a regular generator" score. The single change is to the regular generators people entertain: besides a switch-biased and a heads-biased coin, they also consider a repeating-motif process that copies a short pattern (one to four flips) with an occasional slip, so near-periodic sequences such as perfect alternation or HHT-HHT are recognised as regular and look less random.
- **bayesian_chance_vs_motif_beta_trick_coin**: Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with each generator's unknowns averaged out. The single change is in what people imagine a "biased coin" to be: instead of a uniform prior over its heads rate, they hold a symmetric prior of fitted concentration — imagining either only grossly lopsided trick coins or mildly biased ones — which sets how much a modest heads/tails imbalance counts as evidence against chance, addressing the critique that people weight heads/tails balance more than the incumbent's biased-coin generator implies among sequences with similar switch counts.
- **hot_hand_only_suspicion**: People judge a sequence random to the extent a plain fair coin explains it better than the only kinds of "non-random" coin they ever suspect — a hot-hand (streaky) coin that tends to repeat its last outcome by an unknown amount, or a coin biased towards heads or tails by an unknown amount — with those unknowns averaged out and the evidence weighed per flip. Switching is never suspicious in itself, so perfect alternation and period-3/4 patterns count as maximally random: this is where the model disagrees most sharply with the current best model, which condemns them as repeating motifs, while among sequences with similar switch counts it still condemns streaks and heads/tails imbalance.
- **mismatch_noise_paired_comparison**: People do not evaluate each sequence on its own and then compare two independent impressions; they compare the two sequences flip against flip, so positions where the two show the same outcome cancel out and only the mismatching positions carry evidence — and also noise — into the judgment. Each sequence's randomness evidence is the Bayesian "fair coin (believed to over-alternate) versus regular generator" score, but the noise in comparing them grows with the number of positions at which the two sequences differ, so the same difference in randomness is judged decisively beside a near-identical partner (and on short pairs) and hesitantly beside a very different one.
- **bayesian_chance_vs_repeating_motif**: Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators, with every generator's unknowns averaged out normatively. The single change is one more regular generator in the comparison — a "repeating-pattern" generator that picks a short motif (one to four flips) and repeats it — given a fitted share of the prior on regularity; so exactly periodic sequences (perfect alternation HTHT…, but also HHTHHT…, HHTTHHTT…) are recognised as patterns and condemned, addressing the critique that the incumbent over-credits perfect alternation and under-penalises period-3/4 motifs.
- **gist_tail_probability_test**: People judge randomness like an intuitive significance test on a sequence's gist rather than by comparing explanations: they summarise each sequence by its head count, its number of switches and its longest run, and ask how often a fair coin (which they believe switches somewhat more than half the time) would produce a gist at least as rare as this one. A sequence looks random to the extent that this tail probability is high — a typical gist is unremarkable, while a rare gist (a lopsided count, a long streak, or too-perfect alternation) "rejects chance" — and people differ in how decisively this felt surprise drives their choice.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| length_normalised_chance_vs_motif ←selected | 0.00 | 0.00 | — (best) | 0.733 | yes |
| bayesian_chance_vs_repeating_motif_2 | 7.86 | 3.00 | yes | 0.000 | yes |
| mismatch_noise_repeating_motif | 8.62 | 3.60 | yes | 0.000 | yes |
| bayesian_chance_vs_motif_beta_trick_coin | 8.75 | 2.92 | yes | 0.000 | yes |
| hot_hand_only_suspicion | 10.05 | 6.94 | no (within ~2·dse) | 0.261 | yes |
| bayesian_chance_vs_repeating_motif | 21.14 | 4.35 | yes | 0.000 | yes |
| gist_tail_probability_test | 21.42 | 6.08 | yes | 0.006 | yes |
| mismatch_noise_paired_comparison | 26.46 | 6.49 | yes | 0.000 | yes |
