# Inner Loop — round 1, candidate 1 of 6 (exploratory slot)

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

Responses CSV: `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv`
Columns in the responses CSV: `sequence_a,sequence_b,participant_id,trial_index,chose_left`

There are **no feature columns** in this CSV — only the raw H/T sequence strings and the response (`chose_left`). The only numeric column you can read directly as a `pm.Data` is `chose_left`.

Your model **must** compute its own features from the raw sequences. Define a module-level hook in `candidate.py` — either:

- `compute_features(sequence_a: str, sequence_b: str) -> dict[str, float]`: returns named numeric features for one stimulus pair; the pipeline calls it per trial and exposes each key as a `pm.Data` column.
- `prepare_observed(rows: list[dict]) -> dict[str, np.ndarray]`: builds all observed arrays at once from the full row list.

One of these hooks is **required** — without it the model cannot bind any stimulus input.

**Allowed imports:** your `candidate.py` may only import from this allowlist: `arviz`, `collections`, `dataclasses`, `functools`, `itertools`, `math`, `numpy`, `operator`, `pymc`, `pytensor`, `re`, `scipy`, `statistics`, `typing`. Any other import (including the project's feature library, pandas, or any `src.*` module) causes immediate rejection at admission. Every helper your model needs must be written in the file itself — self-contained code only. It may not read files or reach the interpreter either (`open`, `np.load`, `eval`, `__import__` and the like are rejected).

Work in three steps:
1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.
2. Write `model_name.txt` — a short snake_case name for the model (it
   becomes the model's identifier everywhere downstream).
3. Write `candidate.py` — a module-level PyMC model implementing only that
   hypothesis.

**Check your model before you finish.** Run this from the repository checkout (your shell's working directory). It runs the admission gates you can act on — the import allowlist, a loadable module-level `model: pm.Model`, a finite log-probability on the real responses, the data contract (the observed variable is exactly the responses' `chose_left`, `p_left` has one value per trial and is the probability the Bernoulli likelihood uses), a short MCMC fit (100 draws, 100 tune, 1 chain: a smoke test, not a full production fit) and a finite ELPD-LOO — and prints `OK` or the exact reason admission would reject the file. Fix anything it reports. It does not check novelty against the other models, nor convergence, nor speed: admission's full fit must have almost no divergent transitions, R-hat <= 1.05 and bulk ESS >= 100, and each of its sampling runs must finish within 30 minutes (a model still sampling then is stopped and rejected as too slow). A fit that narrowly fails is already refit once with smaller NUTS steps (target_accept 0.95), and one far from converging is not refit at all, so asking for smaller steps is not a fix: prefer smooth, well-identified parameterisations (non-centred hierarchical or scale parameters, priors that constrain every parameter, no parameters that trade off against each other, no hard thresholds in the likelihood) and a likelihood vectorised over trials.

```bash
/scratch/users/kushinm/auto-psych/outer_loop_live/venv/bin/python -m src.pipelines.inner_loop.check_candidate \
    --candidate-dir /scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/iter_1/candidate_1 \
    --responses /scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv
```

`existing_hypotheses.md` lists the hypotheses already in the model set and
how well each fits. Read it so you propose a *distinct* or *refined*
hypothesis — never a blend of several — under a name not already taken.

`attempted_hypotheses.md` lists the hypotheses tried earlier — in this
experiment or a previous one — that are no longer in the model set,
with what happened to each (pruned after losing by a stated margin, or
rejected at admission, most often as a near-duplicate of a model still
in the set). Do not re-propose any of them unchanged or as a
near-duplicate; a pruned mechanism may come back only with a
substantive change.

`critiques.md` (/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/iter_1/critique/critiques.md) is a posterior-predictive critique of
the current **best** model: the test statistics on which it significantly
fails to reproduce the data, each with the direction of the discrepancy
and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not
confirmatory tests — several are checked per round, so prefer a
discrepancy that survives the FDR (`q ≤ alpha`). Use the strongest such
discrepancy to motivate a single mechanism that would close that gap.
