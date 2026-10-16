# Consolidating the three finals into the leanest justifiable model

PI, 2026-10-16. The goal is the smallest model that keeps what each of the three chains' final
models got right. It is exploratory, not part of the registered test (OSF s48jv). The rule below
is committed before any fit, so the lean model is chosen by a fixed procedure, not by eye.

## 1. The superset

`src/rsa/consolidate/superset.py` is one model whose mechanisms switch on and off. It is plain
JAX, not memo, so every switch is exact.

**The core:** plain RSA (Frank & Goodman, 2012) with a lapse to a uniform guess, 2 parameters.
With nothing switched on, it is the starting model `rsa_l1`; a test checks this exactly.

**The 13 mechanisms**, 14 parameters (16 with the core):

| mechanism | what it does | from | parameters |
|---|---|---|---|
| `depth2` | a second level of recursion | chains 0, 2 | 0 |
| `dup` | duplicated objects are less likely referents (log of copies) | all three | 1 |
| `richness` | prior on feature count: linear, plus a U-shape over the display's range (extremity) | all three, in three forms | 2 |
| `empty` | prior bonus for an object with no features | chains 1, 2 | 1 |
| `surprisal` | prior on summed feature surprisal | chain 1 | 1 |
| `redundancy` | synonyms count as k^-w alternatives | chain 2 | 1 |
| `aspect` | a speaker who sometimes names the sparsest match's feature without discriminating | chain 1 | 1 |
| `interior` | speaker penalty for objects inside a word's feature-count range | chain 0 | 1 |
| `avoidable` | speaker penalty on an ambiguous word when a unique one existed | chain 0 | 1 |
| `competition` | lower speaker precision for objects with more features | chain 2 | 1 |
| `confusion` | the literal listener confuses similar objects (lambda 0 is plain RSA) | chain 1 | 1 |
| `binding` | illusory feature binding onto non-matching objects | chain 2 | 1 |
| `attention` | inattention growing with display size, captured by the prior over matching objects | chain 0 | 2 |

**Left out:**
- manipulations the live displays never varied: valence, greyscale, familiarization;
- nuisance terms specific to one chain: chain 2's listener exponent and word prevalence, chain
  0's typicality, solitary foil and inhibition leak.

**Tests** (`tests/test_rsa_consolidate.py`):
- every mechanism at its "off" value reproduces the model without it;
- every mechanism, switched on, changes some display's prediction;
- every variant meets the model contract.

## 2. The rule (fixed 2026-10-16, before any fit)

**Data:** the campaign's pooled data only (`finals_pool/pooled.csv`, 70,841 trials: the existing
plain trials and the 9 live experiments, catch trials removed). This is the data the frozen
predictions were fitted to. The pre-registered experiment's data are not read.

**Criterion:** ELPD from grouped 5-fold CV (`src.rsa.loop.cv`):
- folds of whole displays and conditions, balanced by source, seed 0;
- the loop's fit rule: dense NUTS, 4 × (1,000 + 1,000), one refit at `target_accept` 0.95;
- the SE of a difference is clustered by unit.

**Backward elimination** (`src.rsa.consolidate.eliminate`):
1. Start with all 13 mechanisms on.
2. Each round, score the current model and each model with one mechanism off.
3. Take the converged one-off model with the highest ELPD-CV. Drop its mechanism if the cost (the
   current model's ELPD-CV minus its own) is ≤ 0 or below **2 clustered SEs**.
4. Stop when every removal costs ≥ 2 SE. The current model is then the lean model.

No other model is considered, and nothing is added back.

## 3. The prospective check

`src.rsa.consolidate.score` fits each model on the same pooled data, then scores it on the
pre-registered experiment's 14,404 modelled trials, never fitted to:
- the lean model;
- the full superset;
- the three finals;
- the best promoted seed.

**The comparison:** each model against chain 0's final, the strongest model in the registered
test (17 parameters), by log predictive density, with SEs clustered by display, overall and per
half of the design.

**The reading, stated now:**
- the lean model **consolidates** if it scores within 2 SE of chain 0's final, or better, with
  fewer parameters;
- it **falls short** if it is more than 2 SE worse. That would mean something chain 0's final
  has is missing from the superset.

## 4. Running it on Sherlock

1. Pull `auto-rsa`.
2. Run `STAGE=eliminate sbatch scripts/rsa/slurm/consolidate.sbatch`.
   - Work root: `$SCRATCH/auto-psych/rsa_consolidate`.
   - It copies `rsa_prereg/pool/{pooled.csv,.fit_cache}` once.
   - 24 cores (all of the mcfrank node), 7-day limit. A resubmit resumes: every model's CV is
     saved. Each stage stages its own copy of the code (`code_<stage>/`; fix 393f3f2c).
   - **Expected size:** greedy backward elimination, not all subsets. Round 1 scores 14 models
     × 5 folds (70 fits), each later round one model fewer, and it stops at the first costly
     removal (at most 455 fits). About 20 fits run at once, so roughly 13–33 h in all.
   - Submitted as job 48051947 (2026-10-16).
3. Run `STAGE=score sbatch scripts/rsa/slurm/consolidate.sbatch`.
   - It fits and scores the lean model and the comparison models on the pre-registered data
     (`rsa_prereg/cells/prereg/run/experiment1/data/modelled.csv`).
4. **Bring back** to `data/rsa/consolidate/` and push:
   - `eliminate/elimination.json`;
   - `eliminate/lean/` (the lean model file);
   - `score.json`.

   These are sums only; no trial data.
