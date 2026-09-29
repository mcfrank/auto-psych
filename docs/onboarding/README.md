# Onboarding: auto-psych in one hour

These pages are for a collaborator who knows the science (Bayesian models of
judgment, online experiments) but has not read the code recently. They were
written on 27 September 2026 against the branch `fix/audit-2026-09-27`, and
every command, flag, default and cost in them was checked against the code on
that date. Where something could not be checked, the text says so.

## What the project does

auto-psych is an automated discovery loop for cognitive models. It currently
studies **subjective randomness**. On each trial a participant sees two coin-flip
sequences of the same length (for example `HHTHT` and `HTTTH`) and clicks the
one that looks more random. Each hypothesis about how people make that judgment
is written as a small Bayesian model in PyMC. The model predicts
`p_left`, the probability of choosing the left sequence.

The loop repeats four steps, one experiment at a time:

1. **Choose stimuli.** From every possible pair of sequences, pick the 64 pairs
   on which the current models disagree most, measured by expected information
   gain.
2. **Run the experiment.** An AI coding agent builds a jsPsych web experiment
   from a fixed template. The pipeline hosts it on Firebase and recruits paid
   participants on Prolific (or uses simulated participants).
3. **Fit, criticise and extend the models.** Every model is fitted to all data
   collected so far by MCMC. A critique agent looks for patterns in the data
   that the current best model misses. Several coding agents then each propose
   a new model, and a proposal is kept only if it fits properly and makes
   genuinely different predictions from the models already there.
4. **Keep the plausible models.** Models that are clearly worse than the best
   one (by cross-validated predictive accuracy) are dropped. The rest carry
   over to the next experiment, where they decide the next stimuli.

```
 experiment 1 starts from 4 literature models ("starting models")
        │
        ▼
 ┌─────────────────────────────── one experiment ──────────────────────────────────┐
 │                                                                                 │
 │  [2_design]          [3_implement]           [4_collect]        [5_model_loop]  │
 │  choose 64 pairs  →  AI agent builds the  →  participants   →   fit all models, │
 │  by expected         jsPsych page; deploy    respond; data      critique, AI    │
 │  information gain    to Firebase; open a     fetched from       agents propose  │
 │  (no AI)             Prolific study          Firebase           new models,     │
 │                                                                 compare, prune  │
 └──────────────────────────────────────────────────────────────────────┬──────────┘
        ▲                                                               │
        └──── surviving models + record of every hypothesis tried ──────┘
                                  (next experiment)

 Final output: the surviving models of the last experiment, ranked, with their
 code, their plain-language hypotheses, fits and comparison tables.
```

The bracketed names (`2_design`, …) are the stage names used by the code and by
the `--agent` flag. There is no stage 1; the numbering is historical.

## Before you deploy anything: known problems

When these pages were written, the live path **could not run a complete
experiment**. Two problems were confirmed by reading the code, and the first was
also reproduced:

- **Fixed on 28 September 2026: the model-fitting stage stopped at its first
  round of new models.** The collected `responses.csv` had extra columns
  (Prolific IDs among them), and the stage refuses any column beyond the five
  it expects. Collection now keeps only those five where agents can read them.
- **The live launchers never load `bubblewrap`.** The coding agents need it to
  run, so the job fails at the experiment-building stage. This one fails
  before anything is deployed or paid for.

The details, a no-cost way to catch both, and a list of smaller surprises are in
[running_a_live_experiment.md § 0](running_a_live_experiment.md#0-read-this-first-known-problems-as-of-27-september-2026).
Get them fixed and the no-cost rehearsal passing before recruiting anyone.

## Reading order

| Page | What it gives you | Time |
|---|---|---|
| [how_the_loop_works.md](how_the_loop_works.md) | The loop step by step, the reasoning behind each design choice, and where the code is | 20 min |
| [running_a_live_experiment.md](running_a_live_experiment.md) | Runbook for live Prolific + Firebase runs: credentials, setup, config, safety gates, no-cost rehearsal, launch, monitoring, stopping, data, privacy | 25 min |
| [simulations_and_validation.md](simulations_and_validation.md) | The "recovery" simulations and "impossible" controls, and how to read them | 5 min |
| [troubleshooting.md](troubleshooting.md) | Error messages you are likely to see, and what to do | skim |
| [glossary.md](glossary.md) | Every technical term and code name used here, in plain words | as needed |

## Where to look in the code

| You want… | Look at |
|---|---|
| The pipeline entry point and all its flags | `src/pipelines/outer_loop/run.py` (`--help`) |
| Stage orchestration, starting models, collection | `src/pipelines/outer_loop/orchestrator.py`, `collect.py` |
| Stimulus choice | `src/pipelines/outer_loop/eig.py`, `src/models/eig_selection.py` |
| Firebase and Prolific | `src/pipelines/outer_loop/deployment/` (`local.py` is the top-level flow), `src/runtime/prolific.py`, `functions/index.js` |
| The model-discovery stage | `src/pipelines/inner_loop/` (`pymc_orchestrator.py`, `model_zoo.py`, `scoring.py`, `candidate_agent.py`, `critique_round.py`) |
| MCMC settings | `src/models/mcmc_defaults.py` (the single source of sampler defaults) |
| The subjective-randomness task and starting models | `src/pipelines/outer_loop/projects/subjective_randomness/` |
| Cluster launchers for live runs | `scripts/outer_loop_live/` |
| Live dashboard / results browser | `src/monitor/`, `src/viewer/` |

The repository's own `README.md`, `CLAUDE.md` and `docs/modeling_loop.md` go
into more depth. They were written during the same automated sessions that
built the code, so they use a lot of shorthand, and some statements in them are
out of date. The places where that matters are listed in the pages here.
