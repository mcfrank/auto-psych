# Inner Loop — round 3, candidate 4 of 6 (refinement slot: the incumbent)

# The task

Each participant sees a few example numbers between 1 and 100, produced by a
hidden computer program, and is asked about other numbers between 1 and 100:
would the program also produce this number? The examples on a trial are its
*set* (one to four numbers, shown in the order given); the number asked about
is the *target*. The participant answers yes or no. There is no correct
answer and no feedback.

Every participant answers about many (set, target) pairs, and the same set is
asked about with many targets.

## The data

Each row of `responses.csv` is one answer:

| Column           | Meaning                                                            |
| ---------------- | ------------------------------------------------------------------ |
| `number_set`     | the example numbers, comma-separated in the order shown ("16,8,64") |
| `target`         | the number asked about, 1-100                                      |
| `rating`         | 1 if the participant answered yes (the program would produce it), else 0 |
| `trial_index`    | the trial's position in that participant's session                 |

## Your job

Responses CSV: `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
Columns in the responses CSV: `number_set,target,trial_index,rating`

There are **no feature columns** in this CSV — only the raw stimulus columns (`number_set`, `target`) and the response (`rating`). Every value reaches your model as the string the CSV holds.

Your model **must** compute its own features from the raw stimulus columns. Define a module-level hook in `candidate.py` — either:

- `compute_features(number_set: str, target: str) -> dict[str, float]`: returns named numeric features for one stimulus; the pipeline calls it per trial with the stimulus columns in that order and exposes each key as a `pm.Data` column.
- `prepare_observed(rows: list[dict]) -> dict[str, np.ndarray]`: builds all observed arrays at once from the full row list.

One of these hooks is **required** — without it the model cannot bind any stimulus input.

**Allowed imports:** your `candidate.py` may only import from this allowlist: `arviz`, `collections`, `dataclasses`, `functools`, `itertools`, `math`, `numpy`, `operator`, `pymc`, `pytensor`, `re`, `scipy`, `statistics`, `typing`. Any other import (including the project's feature library, pandas, or any `src.*` module) causes immediate rejection at admission. Every helper your model needs must be written in the file itself — self-contained code only. It may not read files or reach the interpreter either (`open`, `np.load`, `eval`, `__import__` and the like are rejected).

Work in three steps:
1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.
2. Write `model_name.txt` — a short snake_case name for the model (it
   becomes the model's identifier everywhere downstream).
3. Write `candidate.py` — a module-level PyMC model implementing only that
   hypothesis.

**Check your model before you finish.** Run this from the repository checkout (your shell's working directory). It runs the admission gates you can act on — the import allowlist, a loadable module-level `model: pm.Model`, a finite log-probability on the real responses, the data contract (the observed variable is exactly the responses' `rating`, `p_yes` has one value per trial and is the probability the Bernoulli likelihood uses), a short MCMC fit (100 draws, 100 tune, 1 chain: a smoke test, not a full production fit) and a finite log-likelihood — and prints `OK` or the exact reason admission would reject the file. Fix anything it reports. It does not check novelty against the other models, nor convergence, nor speed: admission's full fit must have almost no divergent transitions, R-hat <= 1.05 and bulk ESS >= 100, and each of its sampling runs must finish within 30 minutes (a model still sampling then is stopped and rejected as too slow). A fit that narrowly fails is already refit once with smaller NUTS steps (target_accept 0.95), and one far from converging is not refit at all, so asking for smaller steps is not a fix: prefer smooth, well-identified parameterisations (non-centred hierarchical or scale parameters, priors that constrain every parameter, no parameters that trade off against each other, no hard thresholds in the likelihood) and a likelihood vectorised over trials.

```bash
/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/venv/bin/python -m src.pipelines.inner_loop.check_candidate \
    --candidate-dir /scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/iter_3/candidate_4_repair_1 \
    --responses /scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --method warm_start_nuts
```

`existing_hypotheses.md` lists the hypotheses already in the model set and
how well each fits. This is a refinement slot — `CANDIDATE_BRIEF.md` says
which model you improve — so your model is a better version of one already
proposed, under a name not already taken.

`refinement_menu.md` lists the models you may draw on — the live models
other than the incumbent and the models pruned earlier — with their full
hypotheses, their standing or the margin by which they lost, and their
source files. Read the source of the model you refine.

`critiques.md` (/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/critiques.md) is a posterior-predictive critique of
the current **best** model: the test statistics on which it significantly
fails to reproduce the data, each with the direction of the discrepancy
and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not
confirmatory tests — several are checked per round, so prefer a
discrepancy that survives the FDR (`q ≤ alpha`). Use the strongest such
discrepancy to motivate a single mechanism that would close that gap.
