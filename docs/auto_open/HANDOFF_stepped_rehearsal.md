# Handoff: the stepped rehearsal, both arms in step mode (2026-11-08)

For the Sherlock session and the PI. Written by Claude Opus 5.5; not reviewed by a person.

**Where this sits** (`REREGISTRATION.md`, "The plan to registration"): (1) **this stepped rehearsal**,
simulated participants, to test step mode with the faster fits and the new theorist instructions; (2)
register, then a first open and a first fixed ("closed") chain behind a feasibility gate; (3) three more
chains per arm. "Pilot" means step 2's first pair (or chain 1, the earlier pilot), not this run.

**Purpose.** Measure what a registered chain will cost, on the code it will run, before the
registration sets its size: each step's wall time and queue waits, agent time against fit time, memory,
and failures, for **both arms**, with real agents and simulated participants (no Prolific, no
participant data spent). Secondarily, see what the new theorist instructions produce.

**Code: `auto-psych-open` at this handoff's commit or later.** Since the last rehearsal:

- fits about 10× faster on large corpora (review V3 §13-14): one evaluation per display, pattern and
  display shape; the sampler reaches the density through a callback compiled once; one start search per
  chain (`HANDOFF_fit_speed.md`, rounds 1-3);
- fits run on one core again (the callback's deadlock fixed with two XLA pool threads, `PJRT_NPROC`), and
  in-process fits free their compiled programs (a memory leak that crashed a slow test);
- **no hand-set numbers** in a theory (`gates.magic_number_problems`, at admission and in the self-check);
- **the one-SE rule** (PI 11-08): of the trusted, eligible theories within 1 clustered SE of the best on
  the newest paradigm, the one with the fewest parameters leads (`corpus.simplest_adequate`). Pruning still
  measures distance from the best fit and never prunes the incumbent. Each scored step records `fit_best`
  (the best by fit alone), `n_params` and `parsimony_k` in `history.json`;
- **the theorist prompt** (`open_theory.md`, "What a good theory is, and how the loop chooses"): the aim
  (a short theory that carries over to new people and paradigms), the one-SE rule, and that simplifying is
  progress; a refinement slot's one change may be a simplification; each theory's standing shows its
  parameter count and why the incumbent leads.

## 1. Before any agent runs (this session, no agents)

1. `HANDOFF_fit_speed.md` round 3: the slow tests (2 CPUs, 32 GB) and the one-core timing of
   `figure_ground_crowd_topic_tom` on experiment 3's corpus. Report after V3 §14.
2. Fast tests: `uv run pytest -q -m "not slow" tests/test_rsa_open*.py`.

If either fails, stop and report; the PI does not launch.

## 2. The launch (the PI, by hand: rule 10)

Both arms, chain 0 of each, in one fresh work root, so they share its ground truth (copy
`private/ground_truth/` from an earlier rehearsal root first, or let the first job build it). Three rounds
of six theorists (the size the registration is leaning to), three experiments set up, one experiment per
launch, so each stops with a mail after experiment 1:

```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_stepped_rehearsal_2 && mkdir -p $WORK_ROOT/logs
for ARM in open fixed; do
  STEP_MODE=1 CHAIN=0 ARM=$ARM N_EXPERIMENTS=3 MAX_ITERATIONS=3 CANDIDATES=6 \
    sbatch --job-name=rsa_open_steprh_${ARM}_c0 --partition=mcfrank --cpus-per-task=8 --mem=32G \
      --time=2-00:00:00 --no-requeue -o $WORK_ROOT/logs/%x_%j.out scripts/rsa/slurm/open_rehearsal.sbatch
done
```

Later steps go to `STEP_PARTITION` (`mcfrank` unless set; `normal` rather than `owners` if the lab node is
busy: a step is never requeued). Launching experiment 2 of either arm is the PI's choice after reading the
report (`CLEAR_STOP=1 STEP_MODE=1 …`, as the STOP file says): it measures a larger corpus, at the cost of
another day.

## 3. What to report (after the mails; no polling)

In a new section of `DISTRIBUTED_DESIGN_REVIEW_V3.md` (or a new file), per arm:

1. **Time.** Launch to mail; per step, its queue wait and run time and what it did (design, agents, fits,
   scoring); per round, agent time against waiting for fits. Agent minutes in total.
2. **Fits.** Every array task's run time, queue wait and MaxRSS, by kind (admission, carried, fold); the
   memory tier each used; how many ran under 20 minutes (SRCC's sensible minimum for a job). Whether any
   fit hung, ran out of memory or hit its time limit, and convergence (first fit, near-miss refits).
3. **The decision this informs.** If most fit tasks are short and their queue waits are a large share of a
   round, say so: the step job could run its own fits instead of submitting arrays. Project a 3-experiment
   chain's wall time from the measurements, and say what it rests on.
4. **Failures.** Any STOP, a step that died, a session run twice, a message from Slurm or SRCC, with job ids.
5. **What the theorists made.** For each candidate: parameters, lines, admitted or refused and why (how many
   refusals were for hand-set numbers, and whether the repair then passed); at each scored step the
   incumbent and `fit_best` and their parameters; whether any refinement was a simplification. For
   comparison, chain 1's experiment-1 theories' parameters and lines.
6. **Cost:** agent tokens by role, core-hours by kind, and `sshare -U -l` before and after.

Aggregates only, as always: no trial-level data or ground-truth names in the report.
