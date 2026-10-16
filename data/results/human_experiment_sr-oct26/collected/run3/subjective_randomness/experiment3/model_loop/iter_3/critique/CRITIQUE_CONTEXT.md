# Critique context

# The task

Each participant takes part in one experiment. On every trial they see two
sequences of coin flips, written as strings of H (heads) and T (tails), side by
side. The two sequences always have the same length, between 2 and 8 flips;
lengths vary from trial to trial. The participant chooses the one sequence that
looks more random to them. There is no correct answer and no feedback.

Before the first trial, participants read these instructions:

> In this study, you will look at sequences of coin flips and judge how
> random they look.
>
> Imagine flipping a fair coin over and over. Each flip is equally likely to
> come up Heads (H) or Tails (T), and every flip is independent — the coin has
> no memory, so what came before does not change what comes next.
>
> On each trial you will see two sequences of coin flips, side by side. The two
> sequences will have the same length. Your task is to pick the one sequence
> that looks more random to you — the one that looks more like it was produced
> by genuinely random coin flipping.
>
> Different people have different impressions of what makes a sequence look
> random, and there are no right or wrong answers. We are interested in your
> own honest impression, so go with your gut.

Each trial asks "Which sequence looks more random?" and the participant clicks
`Left` or `Right`. Which sequence of a pair is shown on the left is randomised.

## The data

Each row of `responses.csv` is one choice:

| Column           | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| `sequence_a`     | the sequence shown on the left                                  |
| `sequence_b`     | the sequence shown on the right                                 |
| `chose_left`     | 1 if the participant chose the left sequence as more random, else 0 |
| `participant_id` | the participant; unique across experiments                      |
| `trial_index`    | the trial's position in that participant's session              |

Every participant in an experiment sees the same set of pairs.

## Your job

**Incumbent (best) model:** `short_sequence_pseudoflip_dilution`
**Incumbent model code:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/short_sequence_pseudoflip_dilution.py`
**Incumbent hypothesis:** Refinement of the incumbent `heavy_tailed_signed_sensitivity_motif` (people judge a sequence random to the extent their own second-order model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a heads-favoured trick coin and a repeating motif with slips — weighing the evidence per flip, with heavy-tailed signed person sensitivity and a side habit). The single change is in how the evidence is weighed per flip: people do not judge a very short sequence on its few flips alone but read its impression as if diluted by a handful of imagined, unremarkable flips (a fitted pseudo-length added to the real length), so the per-flip weighting no longer inflates the evidence of 2- and 3-flip sequences — people still prefer the more-switching sequence in very short pairs, but less decisively than a pure per-flip scaling implies, addressing the critique that in pairs of length 2–3 people choose the more-switching sequence less often than the incumbent predicts.

**Responses CSV:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `sequence_a,sequence_b,participant_id,trial_index,chose_left`
**Model set directory:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models`

Propose **8** test statistics. Write each to `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/iter_3/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/iter_3/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/responses.csv \
    --model short_sequence_pseudoflip_dilution \
    --models-dir /scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models \
    --test-stats-dir /scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/iter_3/critique/test_stats \
    --out /scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/iter_3/critique/ppc_results.json \
    --cache-dir /scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/.fit_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/iter_3/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.
