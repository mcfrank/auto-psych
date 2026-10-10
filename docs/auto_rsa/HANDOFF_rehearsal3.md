# Handoff: rehearsal 3 (short): the critique step and the new design, with real agents

From the driver session, 2026-10-10, for the local Sherlock session. PI decision: the RSA
loop gets main's critique step (CriticAL) so the two phenomena run one framework, and a short
rehearsal checks it with real agents before the campaign. No money is spent: participants
are simulated.

## 0. What is new since rehearsal 2

- **The critique step** (`src/rsa/loop/critique.py`; on in the outer loop,
  `OuterConfig.critique`). It runs before every inner-loop round:
  1. A critique agent (same model, sandbox and no-network rules as the candidates) writes
     test statistics on the trial data.
  2. They are scored against 1,000 datasets simulated from the incumbent's posterior.
  3. The significant discrepancies go into every candidate's brief as `critiques.md`.

  Main's rules apply:
  - 8 statistics;
  - p ≤ 0.05 counts as a discrepancy, with a Benjamini–Hochberg q beside it;
  - one retry;
  - a round with no usable statistic runs without a critique, recorded.

  Each round's `history.json` entry has a `critique` record. Its spend is labelled
  `rsa:critique`.
- **The new design:** quotas with EIG inside them, the wider display pool (1,597 displays),
  and the free design recorded beside it (`PLAN.md`, "Design mixture").
- **Recovery's KL** is now a mean per display (86906717).

## 1. Pull, and a fresh work root with rehearsal 2's ground truth

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa
export WORK_ROOT=$SCRATCH/auto-psych/rsa_rehearsal3
mkdir -p "$WORK_ROOT/logs"
cp "$SCRATCH/auto-psych/rsa_rehearsal2/ground_truth.json" "$WORK_ROOT/"
```

Copy the ground truth. The coherent rule measures distance on the design pool, which has
changed, so recomputing it could pick a different model. With the copy, rehearsal 3 has
rehearsal 2's ground truth (`crowd_discrim_confusion_chromatic_l2`).

## 2. Launch

```bash
N_EXPERIMENTS=1 MAX_ITERATIONS=2 WORK_ROOT=$WORK_ROOT \
  sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" \
  scripts/rsa/slurm/outer_rehearsal.sbatch
```

This is one guarded run on chain 0: one experiment of 200 simulated people × 10 of 40
displays, then an inner loop of 2 rounds × 6 slots. Expect about 5–6 h:
- the design takes about 1.5 h;
- the seeds' fits and CV come next;
- each round is about 1.5–2 h, including a few minutes for the critique.

## 3. Check, and report (`REHEARSAL3_REPORT.md`)

**The critique** (the point of this rehearsal), in `experiment1/model_loop/`:
1. **Did each round have a critique?** Each round's `history.json` entry has
   `critique.status`: `critiqued` is the goal. For `no_critique`, give the `reason`.
2. **What the agent wrote,** per round, in `round_<k>/critique/` (or `critique_retry_1/`):
   - the statistics' names and descriptions;
   - any set aside in `broken_statistics/`, and why: too slow, raised, or not finite;
   - from `ppc_results.json`: `n_evaluated`, `n_significant` and `n_significant_fdr`;
   - the top discrepancies, with observed value, model mean and p.
3. **Did the candidates use it?** Check that `round_<k>/candidate_*/critiques.md` exists,
   and say how many candidates' `hypothesis.md` refer to a discrepancy from it. This is a
   qualitative read.
4. **Time:** how long each critique took, from the `[critique]` log lines to the first
   candidate's start.
5. **Spend:** the `rsa:critique` share in `token_usage_summary.json`.

**The design** (`experiment1/design/eig.json`):
- the quotas met (`designs[0].quotas`: each count ≥ its minimum);
- `kinds`;
- `power` against `free.power`;
- `screened_out` (should be empty).

**As before:**
- the ground truth is in no agent output (`gt_name_mentions.txt`; the critique dirs are
  inside `round_*`);
- the recovery record (KL is now per display);
- anything that crashed.

Bring the outputs back as for rehearsal 2 (`HANDOFF_rehearsal2.md` §4) into
`data/rsa/rehearsal3/`. Leave out trial-level CSVs and fit caches.

## 4. Not affected

The live stage-1 re-run (`STAGE1_FINDINGS.md` §7) and the pilot don't run an inner loop. They
can go ahead before, during or after this rehearsal. The campaign waits for this report.
