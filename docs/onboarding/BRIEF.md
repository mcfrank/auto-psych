# auto-psych: the brief

For a coauthor of the workshop paper who has not looked at the code for a
while and will run the live human experiments. It covers the system in the
paper, how the current loop differs from it, where things stand, and what you
need to do before a live run. It was checked against the branch
`fix/audit-2026-09-27` on 28 September 2026.

## 1. The paper's system in three paragraphs

The task is still subjective randomness: on each trial a participant sees two
H/T sequences and clicks the one that looks more random. Each hypothesis is a
small PyMC program that gives `p_left`, the probability of choosing the left
sequence, with free parameters such as choice sensitivity and side bias.

The **outer loop** runs experiments. In the paper, a theorist agent added a
model, a design agent proposed 100–300 candidate pairs, and the pipeline kept
the pairs with the highest expected information gain (EIG) about which model
in the **model registry** is right. An implementation agent built a jsPsych
page from a fixed template; the pipeline put it on Firebase, recruited 40
people on Prolific and downloaded the data.

The **inner loop** (Box's loop) fitted every model by MCMC and scored it by
ELPD-LOO: leave-one-out predictive accuracy, estimated from the posterior
samples. A critic agent proposed 8 test statistics and compared each on the
real data with data simulated from the best model (the CriticAL method). A
theorist agent then wrote new models aimed at the misfits. After two rounds
the best model joined the registry. The paper showed that the loop recovered
held-out starting models and psychologically "alien" rules in simulation,
that it did worse without the inner loop, and that three human replicates
found models that beat the literature models.

## 2. How the current loop differs

Each item says what changed and why. Code pointers are in
[how_the_loop_works.md](how_the_loop_works.md).

1. **New starting models.** The paper's four "seed" models were loose
   adaptations. They were replaced by closer implementations of the same four
   accounts: `falk_konold_dp` (was encoding compressibility), `motif_stack`
   (Bayesian diagnosticity, Griffiths et al. 2018), `finite_experience_occurrence`
   (window typicality, Hahn & Warren 2009) and `local_representativeness`
   (prototype similarity, Kahneman & Tversky 1972). Beating them therefore
   means more. The paper's discovered winners are not among them. Since 28
   September 2026 they are treated like any other model: removed when the
   data clearly favour another (item 8). Before that they were **protected**
   (never removed); the Gemini simulations still running were started that
   way, so their results must not be pooled with later ones. Each run records
   which rule it used. The literature stays in every comparison through the
   baseline in item 10.

2. **No theorist or design agent in the outer loop; stimuli are chosen by
   the program.** It lists all 43,434 pairs of distinct same-length sequences,
   lengths 2–8, and picks 64 one at a time. Each pick maximises the EIG of
   the *whole set* and counts all N participants' answers to each pair. Once
   another pair adds no more than Monte Carlo noise, the remaining places go
   to the pairs most informative for a single answer. Experiment 1 simulates
   from the models' priors; later experiments first fit every model to all
   data so far. *Why:* an agent's few hundred candidates were an arbitrary
   slice of the space, and scoring pairs one by one ignored that pairs
   overlap and that 40 people answer each. New models now enter only in the
   inner loop. Pairs are always the same length, and a study has 64 trials
   (about 7 minutes) instead of 32.

3. **Models see only the raw responses.** The data have five columns
   (`sequence_a`, `sequence_b`, `participant_id`, `trial_index`,
   `chose_left`), and each model computes its own features from the
   sequences. *Why:* precomputed feature columns limited what a hypothesis
   could express, and extra columns carried things agents must not see: the
   generating model's name in simulations, Prolific IDs in live data.

4. **Several theorists per round, with assigned roles.** The critic runs
   first; its critique goes to every theorist. With C theorists per round
   (C ≥ 4), C − 3 must propose a new mechanism, each from a different one of
   eleven angles (memory and attention, decision rule, similarity to a
   prototype, …); two try to improve the current best model; one improves
   another model of its choice, including removed ones. With 3 (the live
   presets' default) there is one of each. *Why:* in an earlier batch of simulations
   the best model never changed in 27 scoring steps; every proposal was new,
   and nobody refined the leader. A theorist that writes nothing is rerun
   once, and a rejected proposal gets one repair attempt with the rejection
   reason.

5. **A record of every hypothesis tried.** Every proposal (admitted or
   rejected, with the reason) and every removal is logged. The log is carried
   across experiments and shown to theorists as "already tried, do not
   re-propose". *Why:* agents kept re-proposing removed ideas (11 of 13
   re-proposals in one simulated run).

6. **Stricter admission.** In the paper any model with a finite likelihood
   joined. Now a proposal must:
   - pass a code check (allowed imports only, no file access);
   - honour the data contract: its likelihood is Bernoulli on the observed
     choices with exactly its `p_left`, so it cannot be scored on one quantity
     and used for design and evaluation through another;
   - fit within 30 minutes per sampling run;
   - converge: chains agree (R-hat ≤ 1.05), enough effective draws
     (ESS ≥ 100), ≤ 0.1% divergent transitions; a near miss is refitted once;
   - differ from every model already there: root-mean-square difference in
     `p_left` ≥ 0.002 on 512 random pairs drawn separately from the
     experiment's stimuli.

   *Why:* unconverged fits and near-copies distorted the comparisons.

7. **How the best model is picked.** The score is still ELPD-LOO. The best
   model is now the top-ranked one whose leave-one-out estimate is reliable
   (at most 1% of trials fail the estimator's diagnostic), not the argmax of
   the softmax "posterior" over models. That posterior is still reported
   but never used for the choice. *Why:* it is rounded to six decimals, so
   all far-behind models tied at 0, and the argmax then picked by file
   order. In 62 of 230 simulated experiments it exported a starting model
   that was far behind. The posterior also no longer has the paper's
   penalty of 0.05 per code line: in finished runs, code length told
   nothing about which model predicts better.

8. **What is kept and carried over.** The paper carried only the winner. Now,
   at the end of each experiment, any model (starting models included) is
   removed when it trails the best model with a trustworthy score by more
   than two standard errors of the ELPD difference.
   That standard error treats all answers to the same pair as one cluster;
   they are correlated, and the per-answer version is about half as large as
   it should be. The best model is never removed. At most 8 models are kept,
   and *all* survivors carry over.
   The registry gives each equal prior weight in the next design. *Why:*
   rivals that the data could not yet separate were dropped (19 at 40
   experiment boundaries), and the next design should target exactly them.
   Earlier registries used stacking weights, which made EIG zero in 15 of 40
   designs.

9. **Safeguards.** The loop stops with an error rather than guessing. Agents
   run in a sandbox: the code is read-only, they write only their own folder,
   and they get no credentials except their own language-model login.
   Prolific IDs are kept in a separate file that no agent is given.
   Participant numbers are unique across a run's experiments. A crashed stage
   restarts cleanly. An agent that hits the account's usage or rate limit
   waits for the reset (up to 12 hours) and runs again, instead of counting
   as a failed proposal. Claude agents are billed by subscription or by API,
   stated for each run and never guessed. For live runs:
   - `live` mode needs two separate confirmations and a typed `yes`;
   - one number sets both the design's N and the places recruited;
   - a relaunch cannot publish a second study for the same experiment;
   - the page is deployed before the study is created, so a failed deploy
     leaves no study behind;
   - a collection that gives up after 3 hours pauses its study;
   - the cost summary covers Prolific only and says so.

10. **Simulated recovery checks and controls.** The paper's checks remain,
    with the new starting models:
    - **Held-out recovery.** One starting model generates responses for 40
      simulated people per experiment, 3 experiments, 5 rounds of 6
      proposals. The loop starts from the other three, and the agents cannot
      see the hidden model.
    - **Scoring.** The loop's best model is compared with the hidden one on
      about 43,000 pairs not used in training. It must beat the remaining
      starting models refitted to the same data at the end of each
      experiment (removed or not); otherwise discovery added nothing.
    - **Impossible controls** (the paper's "alien" rules) use exactly the
      same settings; recovering them well would mean the loop fits anything.
    - **No-inner-loop variants** (the paper's ablation) still exist.

## 3. Where things stand (28 September 2026)

**Tested.** The fast test suite passed on 28 September (2,052 tests; one
file of 18 tests was left out because it needs a library missing from this
environment). Every live-path fix was tested with Prolific, Firebase and HTTP
mocked.

**Simulation results: partial.** The only numbers so far come from a sweep
with Claude Opus 5.5 as the agents' model, scored mid-run on 28 September.
Each cell compares the loop's best model with the best-fitting starting
model, on held-out pairs:

| hidden model | point in the run | loop RMSE / r | starting models RMSE / r |
|---|---|---|---|
| `falk_konold_dp` | experiment 2, round 5 | 0.013 / 0.999 | 0.049 / 0.985 |
| `motif_stack` | start of experiment 2 | 0.110 / 0.92 | 0.156 / 0.84 |
| `finite_experience_occurrence` | experiment 1, round 5 | 0.030 / 0.94 | 0.088 / 0.54 |
| `local_representativeness` | start of experiment 2 | 0.087 / 0.82 | 0.129 / 0.44 |

So far the loop beats the refitted literature models in all four cells, but
there is one repeat per hidden model and only one cell has finished (final
r 0.999, RMSE 0.020). The Gemini sweep and the impossible controls had no
finished cells when this was written: **results pending.** All of these ran
with protected starting models (item 1).

**Known limitations.**
- **No live run on this code.** Nothing has yet been run against Prolific or
  Firebase since the fixes. Do the no-cost rehearsal first.
- **Use `full_run.yaml` for the real runs.** It models exactly as the
  simulations do (5 rounds × 6 proposals, 30-minute agents, `target_accept`
  0.8, 16 CPUs), except that it takes more MCMC draws; a test keeps the two in
  step. `pilot.yaml` (2 × 3) and `hero_run.yaml` (4 × 7) do not, so the
  simulations do not speak to them.
- A study keeps recruiting if the job crashes or you `scancel` it; only the
  normal 3-hour give-up pauses it.
- People who finish after a pause or after the download are paid (payment is
  automatic) but not modelled.
- Language-model costs are not estimated, and the length of the model stage
  is not known (the config comment says 12–15 hours for 3 experiments).
- Stimulus selection is a fast approximation to exact greedy search, checked
  against it on two designs.
- The paper's human results came from the old system and have not been
  re-run.

**Open questions.** Does the loop now change its best model? Do the impossible
controls fail as they should? Does removing starting models change what the
loop finds? How long and how costly is a live experiment?

## 4. What you need to do for a live experiment

Each step is in the [runbook](running_a_live_experiment.md).

1. Confirm the IRB protocol covers the consent text in `templates/consent.txt`,
   64 trials, about 7 minutes and your pay.
2. Put the four credentials in `$REPO/.secrets`: `PROLIFIC_API_TOKEN`,
   `FIREBASE_TOKEN`, `AUTO_PSYCH_RESULTS_TOKEN`, `GOOGLE_API_KEY` (with
   Claude agents: `claude_auth` in the config and its credential instead of
   the last).
3. Build the environment once with `setup.sbatch`.
4. Copy `pilot.yaml`, set a new `run_label`, participants, pay and modelling
   settings.
5. Do the no-cost rehearsals, including the simulated end-to-end run.
6. Launch with `run_pilot.sh`, read the summary, type `yes`. The launcher
   records your checkout's commit in the run's copy of the code.
7. Watch the live dashboard and the Prolific dashboard.
8. To stop: the Prolific study first, then `scancel`. To recover:
   `RESUME_AGENTS`, never a plain relaunch.
9. Copy results out with `collect_results.sh`, which removes Prolific IDs;
   keep the raw files in access-controlled storage.

## 5. Read further

- [how_the_loop_works.md](how_the_loop_works.md): the loop step by step, with code pointers.
- [running_a_live_experiment.md](running_a_live_experiment.md): the runbook.
- [simulations_and_validation.md](simulations_and_validation.md): the simulated checks and how to read them.
- [troubleshooting.md](troubleshooting.md): error messages and what to do.
- [glossary.md](glossary.md): terms, including code names you will see in files.
