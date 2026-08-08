# Design rationale — experiment 1 (agent `2_design`, `--design-mode agent`)

Four candidate cognitive models diverge on which of a pair's features drive
`p_left`. The registry carries **no per-model weights** (`theories: {}`, only
`reserved_for_new: 0.25`), so EIG was scored from a **uniform prior over the
four models**, using each model's **prior-predictive** `p_left` (experiment 1,
no prior responses). Scored with `--featurize`
(`src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py`).

## Candidate pool

A hand-built, disagreement-targeted pool of **7,644 unique pairs**
(`design/candidates.json`), lengths 2–8, organised into contrast families
(generator reasoning in `gen_design_candidates.py` per-family comments):

- **BALANCE** — balanced vs imbalanced at matched length & alternation. The
  balance-preferring models (minkowski wants `h/n≈0.5`; messy balances
  `imbalance` against `alt`) are set against **artificial_balance**, which
  explicitly penalizes contrived `h≈n/2` sequences via its "balanced"
  generator — the strongest predicted sign flip.
- **ALTERNATION** — high- vs low-alternation at matched length & balance.
  Carves the asymmetric quadratic `alt` penalty (messy & per_run) against
  minkowski's symmetric Minkowski-power penalty, and cleanly separates
  **per_item** (messy: `× n`) from **per_run** (evidence_accumulation_per_run:
  `× (alts+1)`).
- **RUNSTRUCT** — pairs sharing the same `(n, h, alts)` (hence identical under
  minkowski/messy/per_run) but differing motif/run layouts, isolating
  **artificial_balance**'s `rep_motifs`/`alt_motifs` features.
- **LENGTH** — near-ideal long vs short and deviant long vs short, tapping the
  length-accumulation distinction between minkowski/messy (scale × `n`) and
  per_run (scale × runs).
- **PERIODIC** — exact periodic/alternating templates vs messy near-ideal,
  for the artificial-balance and run-count stories.
- **BROAD** — same- and cross-length random pairs as a coverage fallback.

## Selected stimuli

`design/stimuli.json` holds the **32** highest-EIG pairs
(`--top 32`), all carrying `sequence_a`, `sequence_b`, and `eig`.

- **Count:** 32 (matches the 32-trial design constraint).
- **EIG range:** 0.3580 – 0.4066 bits.
- **All nonzero:** every stimulus `eig > 0`; max 0.4066.

The greedy tops skew toward **run-heavy / imbalanced long-vs-short** contrasts
(e.g. `HHTHHHHH|HH`, `THTTTTTT|HTTT`). This is expected: those contrasts sit in
the region where artificial_balance (rewards imbalance, penalizes contrition)
sign-flips against minkowski/messy (reward balance & length) and per_run
(penalize run-poor patterns), so the uniform prior over the four models finds
them maximally discriminative. Balanced-vs-imbalanced and length-mismatch
pairs are well represented in the top 32, confirming the BALANCE and LENGTH
targeting paid off; ALTERNATION/RUNSTRUCT/PERIODIC families form the mid-tail.
