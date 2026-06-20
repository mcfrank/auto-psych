# Subjective randomness: problem definition

The domain of inquiry is the subjective perception of randomness by humans judges. We seek an explantory theory of which sequences humans find more or less random. To explore this we use an forced-choice experimental paradigm, in which participants are asked to choose the more random sequence from a pair.

## Task

On each trial, the participant sees **two sequences of coin flips** (H and T) and chooses which sequence looks **more random**. This is a classic paradigm for studying representativeness and alternation biases (e.g. Griffiths, "The Rational Basis of Representativeness").

## Stimulus design schema

- **Stimulus**: A pair of two sequences of H and T. Each sequence is a string (e.g. `"HHT"`, `"HTHTHT"`, `"HHTHTTHT"`). The two sequences in a pair **may have different lengths** (e.g. length 4 vs length 6).
- **Response**: Which sequence was chosen (e.g. "left" / "right").

## Experiment design constraints

- **Total trials per experiment: 32.** At ~5 s per self-paced trial this is roughly
  **3 minutes of judgments** (≈4 minutes including consent, instructions, and
  debrief). The ~5 s/trial rate is grounded in the closest prior paradigm: Reimers,
  Donkin & Le Pelley (2018) ran a 2AFC short-H/T-string button task of **120 trials
  in ~10 minutes** (≈5 s/trial including instruction reading).
- **Sequence length: 2 to 8** (inclusive).

The default design (`--design-mode exhaustive`) enumerates **every** distinct H/T
pair over lengths 2–8 (≈129k pairs) and greedily selects the 32 pairs that
jointly carry the most information about which model is correct — a diverse set
spread across distinctions, not 32 near-duplicates of the single highest-EIG
contrast.

## Experiment presentation (reproduce VERBATIM — identical across all experiments)

The implemented jsPsych experiment must be identical in every experiment and every
run; **only the stimuli change.** Use the wording below exactly (do not
paraphrase), **button responses only**, and the data contract above (`chose_left`,
`sequence_a`, `sequence_b`).

> **Rendering note (the wording below is Markdown — convert it to HTML).** The
> `**bold**` spans must render as `<strong>` in the experiment; participants must
> never see literal asterisks. Put each paragraph in its own `<p>`, and wrap the
> instructions and debrief in the skeleton's `<div class="auto-psych-prose">`
> (constrained-width, left-aligned) so the text does not span the full screen.

- **Instructions** (first screen, button labelled `Begin`) — use this wording exactly:
  > In this study, you will look at sequences of coin flips and judge how
  > **random** they look.
  >
  > Imagine flipping a fair coin over and over. Each flip is equally likely to come
  > up Heads (H) or Tails (T), and every flip is independent — the coin has no
  > memory, so what came before does not change what comes next.
  >
  > On each trial you will see **two** sequences of coin flips, side by side. The
  > two sequences may be **different lengths**. Your task is to pick the **one
  > sequence that looks more random** to you — the one that looks more like it was
  > produced by genuinely random coin flipping.
  >
  > Different people have different impressions of what makes a sequence look
  > random, and there are no right or wrong answers. We are interested in your own
  > honest impression, so go with your gut. You will complete **32 trials**, which
  > takes about 4 minutes. Your responses are anonymous.

  > **Wording rationale (not shown to participants).** This instruction is grounded
  > in the subjective-randomness literature, which warns that naming the structural
  > cues under study (run length / "streakiness", alternation / "switching", H–T
  > balance) coaches participants and confounds the measurement (Nickerson, 2002,
  > *Psychol. Rev.*; Reimers, Donkin & Le Pelley, 2018, *Cognition*). The accepted
  > practice is to define the generating process — a *fair coin*, 50/50,
  > independent flips — but leave the meaning of "random-looking" for the
  > participant to supply from their own intuition (Williams & Griffiths, 2013,
  > *JEP:General*, Exp. 3 nested condition; Griffiths, Daniels, Austerweil &
  > Tenenbaum, 2018, *Cognitive Psychology*). The question phrasing "which … looks
  > more random" follows the canonical 2AFC framing (Griffiths & Tenenbaum, 2003;
  > Bar-Hillel & Wagenaar, 1991, "which … is most like a coin?"). The earlier draft
  > of this instruction named "too streaky / too regular in switching" examples —
  > exactly the run-length and alternation cues being measured — and was removed for
  > this reason.

- **Per-trial choice prompt** (shown above the two sequences):
  > Which sequence looks more random?

- **Display**: the two sequences side by side, **left = `sequence_a`, right =
  `sequence_b`**, in a monospace font.

- **Choice buttons**: exactly two — `Left` (records `chose_left = 1`) and `Right`
  (records `chose_left = 0`).

- **Trials run back-to-back: do NOT show a fixation cross, blank screen, or any
  inter-trial screen between trials.** The next pair appears immediately after a
  response (a small `post_trial_gap` of a few hundred ms is fine; no fixation).

- **Debrief** (final screen, button labelled `Finish`):
  > Thank you for participating! Your responses help us understand how people
  > perceive randomness in sequences of coin flips. When you click **Finish**,
  > your responses will be submitted and you will be **redirected back to Prolific**
  > to complete the study and receive your completion code.

Do **not** add a consent screen — the deployment injects the IRB consent gate as
the first page automatically.

## Models

Each model is a **PyMC model** in `cognitive_models/<name>.py`. At module load
the file builds, at top level, `with pm.Model() as model: ...` containing:

- One `pm.Data` container per stimulus feature the theory uses (see the
  feature-column list below). Each container must have a **1-element placeholder**
  of the correct dtype (e.g. `np.zeros(1, dtype="int64")`); the pipeline calls
  `pm.set_data(...)` to swap in real data before sampling.
- Priors over the theory's free cognitive parameters (e.g. softmax temperature,
  Beta prior on bias). MCMC infers these — they are **not** hyperparameters.
- A `pm.Bernoulli("response", p=p_left, observed=chose_left)` likelihood, where
  the `observed=` argument is the **exact** `chose_left = pm.Data("chose_left", ...)`
  tensor (not a derived copy).
- A `pm.Deterministic("p_left", ...)` exposing per-trial P(chose_left=1).

The pipeline fits each model on the preprocessed responses, scores it by
`arviz.loo` (ELPD-LOO), and uses posterior-predictive samples for correlations
and posterior predictive checks. No callable-style `def model_name(stimulus,
response_options)` functions — that is the old contract and is no longer used.

## Preprocessed data schema

`model_loop/responses.csv` is produced by the project's `preprocess.py` helper.
It carries the raw columns (`participant_id`, `trial_index`, `sequence_a`,
`sequence_b`, `chose_left`) **plus** the following numeric feature columns, one
per sequence (`_a` and `_b`). Use these names verbatim in your `pm.Data(...)`
containers.

| Column           | Type  | Meaning                                          |
| ---------------- | ----- | ------------------------------------------------ |
| `n_a`, `n_b`     | int   | Length of the sequence                           |
| `h_a`, `h_b`     | int   | Number of H's                                    |
| `p_a`, `p_b`     | float | Head proportion (`h / n`)                        |
| `alts_a`, `alts_b`     | int   | Alternation count (transitions H↔T)        |
| `p_alts_a`, `p_alts_b` | float | Alternation proportion (`alts / (n-1)`)    |
| `max_run_a`, `max_run_b` | int | Longest constant-character run                |
| `max_run_norm_a`, `max_run_norm_b` | float | Longest run scaled to `[0, 1]`      |
| `imbalance_a`, `imbalance_b` | float | Distance from balanced H/T counts          |
| `periodicity_a`, `periodicity_b` | float | Match to a short repeating template       |
| `rep_motifs_a`, `rep_motifs_b` | int | Repetition motifs (n1: maximal constant runs) in the Falk & Konold parse |
| `alt_motifs_a`, `alt_motifs_b` | int | Alternation motifs (n2: maximal alternating sub-sequences) in the parse |

And the observed response: `chose_left` ∈ {0, 1} — 1 means the participant
chose sequence A. Use `chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))`
in every model and pass it to `pm.Bernoulli(..., observed=chose_left)`.

You may pick any subset of these feature columns per theory. You do not need
to use them all — only the ones the theory commits to.

## Running the active outer loop

```bash
# 1. Preprocess a raw responses CSV (adds the feature columns above)
uv run python scripts/subjective_randomness/preprocess.py \
    --input-csv data/subjective_randomness/experiment1/responses.csv \
    --output-csv data/subjective_randomness/responses.csv

# 2. Run the outer loop (theory → design → collect → model loop)
uv run python -m src.pipelines.outer_loop.run \
    --project subjective_randomness \
    --experiment 1 \
    --ground-truth-model length_sensitive_alternation
```
