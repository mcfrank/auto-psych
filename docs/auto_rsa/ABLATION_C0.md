# Ablating chain 0's final: how much of the best RSA model the data need

2026-10-28. Replaces the consolidation's superset elimination (`CONSOLIDATION.md` §6): the
superset's posterior has two modes of comparable mass on some CV folds, so its elimination
cannot start. Following the sr-oct26 ablation (`docs/sr_ablation.md`), the ablation starts from
one final instead: the best-fitting of the three, and removes its own mechanisms. Decided before
any fit (user decisions, 2026-10-28):

- **The final:** chain 0's, `boundary_avoidable_ambiguity_listener` (17 parameters), the best of
  the three finals on the pre-registered experiment (−9,949.6 lpd against −10,086.8 and
  −10,139.4; `PREREG_RESULT_REPORT.md`).
- **The data:** the campaign's pooled data (70,841 trials), the consolidation's. Not the
  pre-registered experiment's: it chose this final, has one framing (so a framing-dependent
  mechanism could not act on it) and a fifth of the trials.
- **The rule:** the sr-oct26 ablation's, unchanged (`src.ablation.eliminate.eliminate`): only
  removals that leave a valid model are candidates; of the converged ones the cheapest goes while
  it costs less than 2 SEs; the starting model must converge on every fold.
- **The CV and SEs:** the consolidation's: 5 folds of whole displays and conditions, seed 0
  (`src.rsa.loop.cv`), and its clusters for the SE of a difference.
- **The fits:** numpyro, dense NUTS, 4 × (1,000 + 1,000), one refit at target_accept 0.95, every
  chain started at the best optimum with the mode guard (`init="map"`, `CONSOLIDATION.md` §6).

Before any fit, the final's mode table on the pooled data and on each fold: 19–20 of 21
optimisations reach one mode; the others lie hundreds of nats lower (no mass). Unlike the
superset, the final has one mode.

## The switchable final

`src/rsa/ablation/final_c0.py` rewrites the final with every mechanism switchable;
`spec_c0.py` lists them (`MECHANISMS`, `REQUIRES`, `PARAMETERS`, `NEUTRAL`). A variant is that
source plus a footer (`src.rsa.ablation.variants`); its name lists what was removed
(`c0_full`, `c0_minus_col_out`).

| mechanism | parameter | removed | needs |
|---|---|---|---|
| `depth2` | | the speaker reasons about L0, not L1 | |
| `interior` | `w_interior` | no interior suppression | |
| `avoidable` | `c_avoid` | no avoidable-ambiguity cost | |
| `confusion` | `w_confusion` | no confusion penalty | |
| `singleton`, `distinct`, `features`, `typicality` | `w_singleton`, `w_distinct`, `w_features`, `w_typical` | that salience term leaves the prior | |
| `valence` | `w_valence` | the feature weight no longer shifts with the framing | |
| `color` | `w_color` | no colour salience | |
| `attention` | `logit_p_base` | the listener always attends | |
| `popout`, `load`, `candidate` | `w_popout`, `w_load`, `w_candidate` | that term leaves the attention gate | `attention` |
| `inhibit_leak` | `w_inhibit_leak` | inattention is captured by matching objects only | `attention` |
| `outlier` | `logit_p_outlier` | no solitary-foil outlier | |

`alpha` and `lapse` are the core: without the lapse (and attention's capture term) an object a
word is false of has probability exactly 0, which a fit refuses.

Two mechanisms can do nothing on these data, so the rule will drop them at no cost; that means
"not tested", not "not needed":
- `color`: no display in the pooled data is greyscale, so it adds one constant to every object's
  prior, which the softmax cancels.
- `popout`: inert in the final itself. It raises attention only when the word fits one object,
  where the attending and the captured inattentive listener choose that same object (no
  prediction moves by more than ~3e-6).

**Tests** (`tests/test_rsa_c0_ablation.py`): with every mechanism on, a variant is the final (the
same priors, and predictions within 1e-6 at four prior draws); each mechanism at its neutral
value is the model without it; each, switched on, moves some prediction (`popout`: none, as
above); the full and core variants meet the model contract; the plan replays the rule over saved
CVs, removes `popout` before `attention`, and stops with an error on an unconverged start.

## Running it

    WORK_ROOT=$SCRATCH/auto-psych/rsa_c0_ablation sbatch scripts/rsa/slurm/ablation_c0_array.sbatch

with `$WORK_ROOT/pool/pooled.csv` the consolidation's pooled data. A driver per round submits the
round's models as one array (one model's 5-fold CV per task) and queues the next driver. The
result: `$WORK_ROOT/ablation/elimination.json`, `summary.md`, `lean/`.
