# Handoff: would cheaper fits change chain 1's decisions? (2026-11-07)

For the Sherlock session. Written by Claude Opus 5.5 at the PI's request; not reviewed by a person.

**Why.** Chain 1 (`chain_0` in the code) has run more than 8 days for 3 experiments, and the PI wants stage-2
chains to take about 3 days. The plan under discussion (driver, 11-07):
- **3 rounds** per experiment (the PI is leaning that way);
- **cheaper full fits:** 4 chains × 500 warm-up + 500 draws, against 1,000 + 1,000 now;
- **B1 folds**, which reuse the full fit's adaptation (your §10 test kept the winner, order and prunes);
- **a carry cap of 6**, against 8;
- **a tighter admission limit**: a lower factor on the median starting fit, and an absolute cap.

This test asks whether the cheaper fits change any decision chain 1 made, measured against how much
the decisions move with nothing but a new seed. It also gathers chain 1's data on the admission limit,
because the PI is considering a tighter limit as a deliberate pressure toward parsimony, and parsimony
(and fit) between arms is what stage 2 is meant to compare.

Like §10: copies of chain 1's files, no agents, no Prolific, nothing written to the chain, aggregates
only (per-trial scores stay on Sherlock). Build on `scripts/rsa/open_compute_checks/` (`opt_tests.py`,
`opt_analyze.py`); fits go to arrays on `owners` (or `mcfrank` when chain 1 is done), single-core.

## The theories and the data

Experiment 3's cumulative corpus (`experiment3/corpus.json` and its folds) and **every theory live at
experiment 3's latest scored step** (its `history.json`), plus each block's reference theory if not
among them. If that is more than about 24, take the 16 best by the latest standing plus the references,
and say so. Record the step and the commit the chain was on.

## The configurations

| | full fit | folds | seed | runs? |
|---|---|---|---|---|
| **C** (chain) | 4 × 1,000 + 1,000 | 300 + 500 from the full fit's medians | chain's | no: the chain's own fits and scores |
| **C1** (noise) | as C | as C | 1 | full fits and folds |
| **H** | 4 × 500 + 500 | as C, from H's full fit | chain's | full fits and folds |
| **H1** (noise at H) | as H | as H | 1 | full fits and folds |
| **HB** | H's full fits | B1: reuse H's adaptation (§10's 100-iteration warm-up) | chain's | folds only |

Everything else is the chain's: dense mass, target_accept, the convergence gate and the near-miss refit
rule. Dense mass and near-miss refits must behave as in the loop: use `loop_fit`, not a bare fit. If C1
costs too much, run it only for the 8 best and the references, and say so.

## What to compare (decided before looking)

Apply the loop's own `standing()` and `end()` rules (as `opt_analyze.py` does) to each configuration's
scores and report, for C, C1, H, H1 and HB:

1. **The incumbent** at the latest step.
2. **The guard** (4 SEs on each block against that block's best): which theories are eligible.
3. **The end-of-experiment prune** at the chain's multiplier, then the carry: the 8 kept, and the 6 a
   cap of 6 would keep.
4. **Trust:** which theories' full fits or folds fail the convergence gate, and how many full fits needed
   the near-miss refit.
5. **Scores:** each theory's out-of-fold lpd by block, the difference from C, and the rank order.
6. **Time and memory:** each fit's warm-up and sampling time and peak, by configuration.

**H is acceptable** if its differences from C are no larger than C1's: the same incumbent, guard verdicts
and carried set wherever C1 agrees with C, and per-theory score differences within C1's spread. **HB on
top of H** is judged the same way against H1. Say plainly where a difference exceeds the noise, and for
which theory.

## The admission limit (no new fits; chain 1's records)

From experiments 1-3's ledgers, fit sidecars and `stage_times.jsonl`:

1. **Fit time against complexity:** for every candidate's admission fit, its time (warm-up and sampling
   separately) against its number of parameters, its memo depth or nesting if that can be read from the
   file, and its out-of-fold score once admitted. Is fit time a usable proxy for complexity here, or
   mostly something else (a hard posterior, a slow model computation)?
2. **What a tighter limit would have rejected:** the limit was 2× the median starting fit (at most 4 h).
   At 1.5× and 1.0×, and with absolute caps of 60 and 90 minutes, which admitted candidates would
   have been rejected, and where did they finish? Was any of them an incumbent or a block's reference?
3. **Repairs after a time-limit rejection:** how many such repairs were there, how many were admitted,
   and did the admitted repair have fewer parameters or a shorter fit than the rejected candidate?
4. **The same at H:** H halves the fits and so the limit, which is relative. Report the projected
   rejections in (2) at H's times too.

## Report

In `DISTRIBUTED_DESIGN_REVIEW_V3.md` §13 (or a new file): the comparison table for (1)-(6), the
acceptability verdicts, the admission-limit analysis, the compute used (core-hours by configuration), and
anything that made a configuration impossible to run as specified.
