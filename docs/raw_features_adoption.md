# Adopting the raw-features change: decision record and merge recipe

*Opened 2026-09-10 while arm C was still running. Written so the decision does
not depend on anyone remembering this week.*

## What the arm tested

`raw_features: true` gives the agents only the two H/T sequences: no
harness-computed feature columns in `data/responses.csv`, and a design scored on
raw rows. Every model must compute the features it uses. Motivation and
mechanics: `docs/raw_features_arm.md`.

The prediction going in was that recovery would **fall**, most for
`falk_konold_dp`, because the provided columns reproduce each ground truth's own
decision variable at R² 0.90–1.00 (exactly 1.000 for `falk_konold_dp`, whose
difficulty predictor is `rep_motifs + 2·alt_motifs`, both supplied).

## What it actually showed (partial: 6 of 12 cells at time of writing)

Paired against the featurized sweep at the same `BASE_SEED`, so repeat *r* is
the same synthetic dataset on both sides:

| ground truth | paired cells | raw | featurized | paired diff |
|---|---|---|---|---|
| falk_konold_dp | 2 | 1.000 | 1.000 | 0.000 |
| finite_experience_occurrence | 1 | 1.000 | 1.000 | 0.000 |
| local_representativeness | 1 | 0.997 | 0.763 | **+0.234** |
| motif_stack | 2 | 0.908 | 0.954 | mixed (run1 +0.021) |

The prediction was wrong in an informative way. `falk_konold_dp` did **not**
fall: agents rederived the motif-parse quantities from the raw strings
(`encoding_difficulty`, `difficulty_predictor`). So the featurizer made that
recovery convenient, not possible — expressibility by supplied columns does not
imply the loop was reading the answer off them. And the hardest ground truth,
`local_representativeness`, improved sharply in its one finished cell, which is
consistent with the ready-made `multiscale_imbalance` column having been an
attractor toward a near-miss.

Regenerate the table with:

```bash
uv run python scripts/subjective_randomness/compare_raw_features_arm.py \
    --raw $SCRATCH/auto-psych/holdout_raw_features \
    --featurized $SCRATCH/auto-psych/recovery_improvement/recovery_2026_09_07/iter2/sweep
```

## Gate before adopting

1. All 12 arm cells finished, and `holdout_raw_features/VERDICT.md` reports every
   check passing — in particular **zero dropped models** and **zero candidates
   importing the featurizer**, or the arm is not a raw-features result at all.
2. `local_representativeness` holds up across its three repeats. In the
   featurized sweep it ranged 0.406–0.991, so one cell at 0.997 can be that same
   variance.
3. `motif_stack` does not degrade systematically once all three repeats are in.

## What to merge, and the three files that conflict

Arm C lives on `arm-c/raw-features` in `$SCRATCH/auto-psych/arm_c/repo`, branched
from the campaign's iteration-2 branch. It therefore carries iterations 1–2's
commits as well (`1f0abae`, `85910cc`, `dfc84a6`, `4f9f73e`, `bd7f032`) — all
user-side or agent work already reviewed, and wanted on `main`.

```bash
ml load system git
cd ~/auto-psych
git fetch $SCRATCH/auto-psych/arm_c/repo arm-c/raw-features:armc
git merge armc          # expect conflicts in exactly three files
```

Both sides changed these three, with **complementary** edits — resolve by keeping
both, not by choosing a side:

| file | `main` added | arm C added |
|---|---|---|
| `src/models/pymc_inference.py` | `MissingStimulusColumns`, `NON_STIMULUS_COLUMNS`, the typed raise | `PROTECTED_ROW_COLUMNS`, `_same_feature_value`, the same-value collision rule |
| `src/pipelines/outer_loop/orchestrator.py` | `screened_out_path=` at both design call sites | `raw_features` parameter and the featurizer-free design |
| `CLAUDE.md` | the screening paragraph | the two-seed-sets paragraph |

Then `uv run pytest -q -m "not slow"`: the baseline is **18 pre-existing
failures** (environmental — missing flask/pydantic/plotnine/requests — plus two
tie-break numerics). Anything beyond that is from the merge.

## What is NOT decided here

Whether `raw_features` becomes the **default** for holdout recovery. Merging
makes it available and maintained; that is separate from switching the default,
which changes what every future recovery number means and deserves the full
12-cell result plus a look at whether the human-data path should match. The
conservative reading of the evidence so far is that raw features are at least as
good and are more principled, which argues for making it the default once the
gate above is met.

## Related

- `docs/raw_features_arm.md` — how the arm works, both seed sets, the two-pool
  trap, and the standing limitation that `features.py` stays importable.
- The campaign journal's operator notes (`$SCRATCH/auto-psych/recovery_improvement/recovery_2026_09_07/journal.md`).
