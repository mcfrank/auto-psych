# Glossary

Statistical terms in a sentence each, and the code names you will meet in
files, flags and logs ("code term" = shorthand from the code, not standard
usage).

## Statistics

- **ELPD-LOO.** Leave-one-out predictive accuracy: for each response, the log
  probability the model gives the actual choice when fitted without it,
  summed. Higher is better; it penalises overfitting.
- **PSIS, Pareto k, "reliable".** ELPD-LOO is estimated by reweighting the
  posterior samples (Pareto-smoothed importance sampling); a per-response
  diagnostic, k, flags when that fails. A score is *reliable* when at most 1%
  of responses fail it.
- **`elpd_diff`, `dse`, `dse_clustered`.** A model's ELPD gap to the best and
  its standard error; the clustered version treats all responses to one pair
  as a group, because they are correlated. Pruning uses it.
- **Convergence.** Chains agree (R-hat ≤ 1.05), enough effective draws (bulk
  ESS ≥ 100), ≤ 0.1% divergent transitions. A **near miss** (code term: ≤ 2%
  divergent, R-hat ≤ 1.2, ESS ≥ 20) is refitted once.
- **EIG.** Expected information gain: the average drop, in bits, in
  uncertainty about which model is true, over data the models simulate.
  **Joint EIG** scores the whole stimulus set with all participants' answers.
- **Noise floor, fill** (code terms). Selection by joint EIG stops once the
  best gain is within Monte Carlo noise; the remaining places are *filled*
  by single-answer EIG.
- **Posterior predictive check / CriticAL.** Compare a statistic of the real
  data with its distribution in data simulated from the fitted model.
  BH-FDR q-values are reported beside the p-values; flags use p ≤ 0.05.
- **Model posterior.** Softmax of ELPD-LOO minus 0.05 per code line;
  overconfident, reported only. **Stacking weights**: reported only.
- **BMA** (`_bma` metrics). The posterior-weighted average of all models'
  predictions.

## The loop (code terms)

- **Stages** `2_design`, `3_implement`, `4_collect`, `5_model_loop`; no stage 1.
- **Seed models** = the four **starting models** (`seed_models/`,
  `starting_models.json`). Removable like any model since 28 September 2026;
  earlier runs kept them always ("protected").
- **Registry** (`model_registry.yaml`): the model prior for the next design,
  equal weights.
- **Candidate**: an agent's proposed model. **Slot**: one agent's place in a
  round; **role**: explore, refine incumbent, refine chosen. **Lens**: an
  explore agent's angle, from `DEFAULT_CANDIDATE_HINTS`.
- **Incumbent**: the current best model.
- **Wave, retry, repair**: agents spawned together; rerun of an agent that
  wrote nothing; rerun of a rejected proposal with its reason.
- **Novelty gate / novelty pool**: the near-duplicate check (RMSE < 0.002)
  on `model_loop/novelty_pool.json`.
- **Zoo** (`model_loop/models/`): the stage's models. **Pruned**: removed to
  `models/pruned/`. **Cap**: at most 8 kept. **Live set / carry-forward**:
  the survivors in `cognitive_models/`, copied to the next experiment.
- **Ledger** (`attempted_hypotheses.jsonl`): every proposal and removal.
- **Screened out** (`design/screened_out.json`): left out of one design.
- **Data contract**: the likelihood is Bernoulli on `chose_left` with the
  model's own `p_left`.
- **Sandbox** (bubblewrap, `bwrap`): what confines each agent.

## Simulations (code terms)

- **Harness**: the code that runs and scores simulations. **Cell**: one
  hidden model × one repeat. **GT** (ground truth): the hidden model.
- **Faithful config**: `holdout_recovery_faithful.yaml`, the standard
  simulation settings.
- **Fitted-seed baseline** (`fitted_baseline`): the remaining starting models
  refitted to the same data.
- **Agent tree**: the trimmed repository copy agents see. **Leak**: any route
  by which the hidden model could reach them.
- **Impossible models**: the paper's "alien" rules.

## Live runs

- **Run label**, **run copy**, **deployment manifest**, **Prolific modes**:
  see the [runbook](running_a_live_experiment.md) § 3 and § 10.
- **`code_provenance.json`**: the commit a run copy came from, written by
  the launcher; the deploy copies it into the manifest.
- **`claude_auth`**: how Claude agents are billed, `subscription` or `api`.
- **Prolific ID** (`participant_id_str`): 24 hex characters; identifying.
