# RSA inner loop, Sherlock run 1: results

Run 2026-10-07 by the local session that followed `HANDOFF_sherlock_run1.md`. Per-cell
outputs are in `data/rsa/sherlock_run1/<cell>/` (handoff §6 file set). Trial-level data,
fit caches and agent logs were left on Sherlock in `$SCRATCH/auto-psych/rsa_run1c/`.

**Headline.**
- **Recovery:** all four recovery cells recovered their withheld ground truth on the
  prediction-distance criterion. The winners are 5-11x closer to it than the best
  starting model.
- **Real data:** results are mixed. One replicate's winner beats the best seed on
  held-out conditions by +83 lpd (2.9 SE). The other's is no better than the seed
  (-7.5, SE 71).
- **Selection problem:** in both real cells, the models that generalise best to held-out
  conditions were pruned by the loop's trial-level PSIS-LOO criterion.
- **Driver changes:** the recovery verdict, the selection criterion and agents' internet
  access all need changes or decisions (§5).
- **Cost:** $313.16 for 220 agent runs.

## 1. What ran

| | |
|---|---|
| Sweep root | `$SCRATCH/auto-psych/rsa_run1c` (**not** the handoff's `rsa_run1`; see §2) |
| Code | `12d4b205c1365706f2c8945557f7500b29b39720` (`code_commit`; the data's `prepared_code` too) |
| Jobs | prepare 46901638; setup 46902401 (COMPLETED, 2 min 37 s); array 46902403 (tasks 0-5, all COMPLETED) |
| Partition | `-p mcfrank` (the lab's owner node sh04-17n29, 24 cores / 192 GB) for every job |
| Array size | 6 tasks x 4 CPUs / 30G / 24:00:00, all six at once (handoff default: 8 CPUs / 32G) |
| Settings | 5 rounds x 6 slots; NUTS 4 x (1000 + 1000); opencode 1.18.35 + Gemini 3.8 Flash, 2,400 s per agent; loop seeds 0 (rep 1) and 1 (rep 2); prune at 2 x clustered dse; at most 8 live models |

The mcfrank node was chosen over the handoff's `normal` (PI instruction: mcfrank first,
then owners). Fairshare on `normal` was 0.31 (effective usage 0.96). 6 x 4 CPUs fills the
node exactly, and the dry runs' fit timings were measured on 4 CPUs.

### Data (`SHA256SUMS`, `SHA256SUMS.recovery`)

```
d4d7fa9ce057439f9e705ef772481f28048c9a5663f2cd88115bca9b62823154  combined_trials.csv
2b80b8493153cc9956a23587e26728a6ef3db8e8397fdb20c090a693c4fb4f72  real/train.csv
a0d44f099166d4d23f95661d9a37dd2c68331050eb46f61376f26b17e710d0d7  real/test.csv
d1dc964d314adfe1d9261b7e59877e1ef9953533e29fab202764893670dfff82  real/split.json
17304ccd2634fbca9c5493a80ac88b077c071dddb910dcaa423c7524cab02e96  recovery_literal/simulated.csv
056b1b405739f7a9a40b7eb4996a37ed60f4ae29be34cbb0369cb52a9b133b4f  recovery_salience/simulated.csv
1a57b3264640fa5f0c95a0b9e0b8e7f17e5132eaf0bd385e18f91d82994b6b7e  recovery_literal/train.csv
22a5cc092b558c38dca2153be4afdf41d93d0902d5318756e1034158e3463ce1  recovery_salience/train.csv
6842d7f1fe1299175d80f695a39afe4d7a7a7a3d810980e584e683a4e322edef  recovery_literal/test.csv
a58d5e57de63cc3002b7f8e2b1747a19ac13e62ec58bc43f0b5c4afa945e1298  recovery_salience/test.csv
f0fb3e5390929a99eac7ef5b55f43a4c4143f5dd09e546d3fc3a6abbc8a0b998  recovery_literal/split.json
15b37b1254fb7791f7de49b337dd2a9f433670cf4b16cfe2c254db971e09acaf  recovery_salience/split.json
19fc8096592b0809b3ef0de1b1460cc1273803b83c0b5833a61223761ce88b17  recovery_literal/provenance.json
3e3c6ea466317df60b82a942f3f243a2a01c71e17844ae82d1554fbb35844cb1  recovery_salience/provenance.json
```

- The trial CSVs are identical across all three prepares (rsa_run1, run1b, run1c) and on
  two code commits. The committed CSVs rebuilt byte-identically every time (the checkout
  stayed clean).
- `split.json` differs between prepares only in its recorded `trials_csv` path.
- The recovery simulations are byte-identical between run1b and run1c, so they are
  deterministic.
- Combined data: 61,084 rows. The test set has 10,632 trials in 84 held-out units.

| source | rows | included | test units | test trials |
|---|---|---|---|---|
| mayn_demberg_2022 | 7,590 | 7,590 | 16/86 | 1,579/7,590 (21%) |
| mayn_demberg_2023 | 15,642 | 15,048 | 31/172 | 3,020/15,048 (20%) |
| mayn_demberg_2026 | 20,196 | 16,764 | 8/43 | 3,556/16,764 (21%) |
| pragmods | 10,168 | 8,569 | 19/87 | 1,549/6,703 (23%) |
| sikos_2021 | 7,488 | 5,625 | 10/34 | 928/4,482 (21%) |

For pragmods and sikos_2021 the split's trial count is below the "included" count. This
is presumably because it counts only forced-choice trials with a display; not checked.

### Checks passed

- Setup:
  - the venv imports jax 0.7.0 on a compute node;
  - the loop CLI takes `--seed`;
  - both recovery datasets have the same held-out units and rows as the real split.
- Every cell's agent tree:
  - withheld seeds were removed (`literal_listener`; `rsa_l1_salience`, `rsa_l1_shared_prior`), with "no file, no manifest entry, no copy";
  - `test.csv`, `split.json`, `provenance.json`, `combined_trials.csv` and `simulated.csv` are absent;
  - `agent tree OK`.
- Round 1 (handoff §3):
  - 44 of 45 round-1 agent slots read PASS/FAIL lines from their self-checks;
  - there are no `opencode/snapshot` directories;
  - MaxRSS was 15-19 GB per cell (of 30G).
- Before committing, the copied outputs were scanned for API keys and Prolific IDs (none
  found). The only CSVs are the aggregate held-out tables.

## 2. Failures and what was done

In order. No agent ran before the final submission, so the failed attempts cost no agent
spend. The final run had no failures.

1. **No opencode >= 1.18.35 on Sherlock.** The module tops out at 1.18.8, so I installed
   it into `$GROUP_HOME/software/npm-global` (`OPENCODE_BIN_DIR`).
2. **prepare 46895861: `uv sync --locked --no-build` failed.** contourpy 1.3.3 has no
   glibc-2.17 wheel. Reran with `VENV_MODE=wheels`, as the handoff advises.
3. **prepare 46896092: every ingest download failed** with `CERTIFICATE_VERIFY_FAILED`.
   uv's standalone Python looks for `/etc/ssl/cert.pem`, which el7 does not have. Fixed by
   exporting `SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt`, which keeps verification on.
4. **setup 46896767: "recovery_literal: train.csv differs from the real train.csv outside
   the choice column".** This was a false alarm:
   - `DataFrame.equals` compares dtypes;
   - `response` and `exclusion_reason` hold text only on rows the loop does not fit, so
     they read as `object` in the real file and as all-empty `float64` in the simulated one;
   - every value was equal.

   Fixed in **19ae7f2** (the check now compares as text). It passes on the real files and
   fails on a copy with one character changed. Array 46896773 was cancelled before it
   started.
5. **Code changed, so a new WORK_ROOT.** `stage_code.sh` refuses a `WORK_ROOT` staged from
   other code. Following the handoff ("to run on new code, use a new WORK_ROOT"), I
   prepared `rsa_run1b` instead of deleting the staged copy.
6. **array 46898755: both real cells died in `check_agent_tree.sh`** with
   `gt_files[@]: unbound variable`:
   - a real cell withholds no seed, so the array of withheld files is empty;
   - el7's bash 4.2 treats an empty array as unbound under `set -u`.

   This was not a leak. Fixed in **12d4b20**, which guards the loops as the other RSA
   scripts already do, and adds a regression test:
   - the test was red under Sherlock's bash 4.2 and green after the fix;
   - `tests/test_rsa_slurm_scripts.py` and `tests/test_rsa_recovery.py`, 76 tests, pass on
     Sherlock;
   - CI's bash 5 cannot reproduce the failure, so the test also checks the source.

   I cancelled the array before any agent ran, and the sweep re-ran in `rsa_run1c`.
7. **GitHub rejected the push of 19ae7f2** with an Internal Server Error (HTTPS and SSH,
   no rules on the branch). I pushed the commit straight to the Sherlock checkout over SSH
   and fast-forwarded, so the hash is unchanged. GitHub accepted it ~30 min later. Both
   fixes are on `origin/auto-rsa`.

`rsa_run1/` and `rsa_run1b/` hold no completed cells and can be deleted.

### Environment the run needed beyond the handoff

Exported in every shell that ran prepare or submit (Slurm passes them on with `--export=ALL`):

```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_run1c
export OPENCODE_BIN_DIR=$GROUP_HOME/software/npm-global/bin
export UV_PROJECT_ENVIRONMENT=$GROUP_HOME/venvs/auto-psych_rsa_run1   # Sherlock rule: envs in $GROUP_HOME
export VENV_MODE=wheels
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export PARTITION=mcfrank CPUS_PER_TASK=4 MEM=30G
ln -sfn $UV_PROJECT_ENVIRONMENT $WORK_ROOT/venv                        # rsa_status.sh hard-codes $WORK_ROOT/venv
```

`prepare_data.sh` ran as an sbatch job on mcfrank (4 CPUs, 16G, 1:30), not on the login
node, because Sherlock's rules put venv builds and data processing in jobs. Compute nodes
reached PyPI and every data host.

## 3. Results per cell

"Held-out vs best seed" is the exported best model's held-out lpd minus the best seed's,
on 10,632 held-out trials, with the stimulus-unit clustered SE.

| cell | exported best | held-out vs best seed | held-out rank | adm / rej / pruned | live at end |
|---|---|---|---|---|---|
| real_rep1 | surprisal_multimodal_l2_listener_2 | **-7.5** (SE 71.3) vs rsa_l2 | 26/40 | 30 / 4 / 31 | 4 |
| real_rep2 | preemption_softmax_listener | **+83.2** (SE 27.4) vs rsa_l2 | 4/40 | 30 / 4 / 34 | 1 |
| recovery_literal_rep1 | softmax_belief_salience_listener | +22.8 (SE 10.1) vs rsa_l1_shared_prior | 8/33 | 25 / 13 / 21 | 8 |
| recovery_literal_rep2 | valence_pragmatic_mixture_listener | +21.6 (SE 10.3) vs rsa_l1_shared_prior | 11/37 | 29 / 6 / 25 | 8 |
| recovery_salience_rep1 | base_literal_focal_lapse_listener | +12.0 (SE 8.5) vs rsa_l2 | 9/36 | 30 / 2 / 25 | 8 |
| recovery_salience_rep2 | distractor_bayesian_salience_listener | +13.4 (SE 8.8) vs rsa_l2 | 3/36 | 30 / 7 / 25 | 8 |

### Recovery

`src.rsa.recovery` gives the exported model's pool RMSE to the ground truth, and its
held-out lpd minus the ground truth's. I applied the same two criteria to every starting
model, using the cells' own fits (§5 item 11 explains why).

| cell | ground truth | exported: RMSE | closest live: RMSE | best seed: RMSE | exported - GT held-out | verdict |
|---|---|---|---|---|---|---|
| recovery_literal_rep1 | literal_listener | 0.0055 | 0.0042 (limited_attention_listener) | 0.048 (rsa_l1_shared_prior) | -1.0 (SE 0.9) | recovered |
| recovery_literal_rep2 | literal_listener | 0.0093 | 0.0001 (bounded_capacity_uniform_prior) | 0.048 (rsa_l1_shared_prior) | -2.2 (SE 2.0) | recovered |
| recovery_salience_rep1 | rsa_l1_salience | 0.0073 | 0.0008 (baserate_salience_listener) | 0.049 (rsa_l2) | -1.5 (SE 1.3) | recovered |
| recovery_salience_rep2 | rsa_l1_salience | 0.0044 | 0.0010 (bayesian_salience_base_rate_listener) | 0.049 (rsa_l2) | -0.1 (SE 0.5) | recovered |

- On RMSE, recovery is real: the winners are at 0.004-0.009 and the seeds at 0.048-0.064.
- In every recovery cell a live model sits closer to the ground truth than the exported
  one. In literal_rep2 it is nearly identical (RMSE 0.0001), but it was not exported. The
  winner there is just under the 0.01 bar.
- In the salience cells the held-out criterion cannot separate the ground truth from the
  seeds. rsa_l1 and rsa_l2 are 14.8 and 13.5 lpd behind it (SE 8.9), within 2 SE, so they
  would pass the verdict's held-out clause.
- In the literal cells the best seeds are 23.8 behind (SE 10.3), failing that clause by
  under 3 lpd.

### Real data: what the selection kept and what generalised

| | real_rep1 | real_rep2 |
|---|---|---|
| best seed on held-out | rsa_l2 | rsa_l2 |
| exported best, held-out vs seed | -7.5 (SE 71.3), rank 26/40 | +83.2 (SE 27.4), rank 4/40 |
| best held-out models | all six top models pruned: +51.7 to +66.7 (SE ~34-37) | top 3 pruned: +89.5 to +93.5 (SE ~36-37) |
| in-sample margin they were pruned by | 177-276 ELPD-LOO (> 2 x clustered dse 51-72) vs the exported model | 152-159 (> 2 x dse 54-57) vs the exported model |

- In real_rep1 the loop's in-sample winner beat the six models that generalise best by
  177-276 ELPD-LOO. On held-out conditions it is 59-74 lpd worse than each of them.
- Per source, real_rep1's winner is close to rsa_l2 except on pragmods (-16.9) and
  sikos_2021 (+12.6).
- real_rep2's winner does generalise. The models it displaced do slightly better on
  held-out data, but within their SE.
- Trial-level PSIS-LOO evaluates the training conditions; it does not test generalisation
  to new conditions (§5 item 12).

Rejections were mostly the novelty gate (prediction RMSE within the 0.002 threshold of
an admitted model): 2-7 per cell. Others:
- non-finite ELPD-LOO: 1 in real_rep1, 3 in literal_rep1;
- one non-converged fit (literal_rep2);
- one empty slot (literal_rep1).

### Cost, time, memory

| cell | agent runs | tokens | cost | wall | MaxRSS |
|---|---|---|---|---|---|
| real_rep1 | 35 | 265.2M | $47.82 | 4:54 | 18.2 GB |
| real_rep2 | 35 | 215.5M | $39.97 | 4:31 | 15.9 GB |
| recovery_literal_rep1 | 43 | 415.6M | $71.43 | 6:44 | 14.9 GB |
| recovery_literal_rep2 | 36 | 361.1M | $58.95 | 6:08 | 14.4 GB |
| recovery_salience_rep1 | 32 | 271.8M | $41.88 | 4:07 | 14.6 GB |
| recovery_salience_rep2 | 39 | 333.9M | $53.11 | 5:29 | 18.2 GB |
| **total** | **220** | **1.86B** | **$313.16** | 08:53-15:41 PDT | |

- No call reported missing usage, so the cost is complete.
- $1.42 per agent run, against smoke test 2's $0.80.
- The sweep's $313 is just over the handoff's $150-300 estimate.
- Most tokens are cache reads (e.g. real_rep1: 232M of 265M).

### Agent activity (`agent_activity.md`, `gt_name_mentions.txt`)

- **Every cell's logs mention arviz and JAX docs URLs.** Most are repeated warning texts
  (the arviz migration-guide notice), not visits.
- **recovery_salience_rep2: 70 GitHub API calls on `langcog/pragmods`, plus raw file
  reads of its `models/` (R code for the RSA models of Frank & Goodman-style reference
  games).** That is close to the literature the salience ground truth comes from (§5
  item 13).
- **recovery_salience_rep1** searched Google Scholar for the Mayn & Demberg and Sikos 2021
  papers. It fetched PDFs from OSF (`9shv3` is Duff, Mayn & Demberg, "The role of
  reinforcement learning in pragmatic reasoning tasks"; `3kefd` is a PsyArXiv preprint),
  escholarship and MIT Press.
- **recovery_literal_rep1** fetched Hawkins et al. 2015 (CogSci).
- **No agent fetched any source dataset.** The two real cells' only GitHub URL is opencode
  downloading ripgrep for itself.
- **"Paths outside the agent's directory" are benign.** They are the opencode binary in
  `$GROUP_HOME` and truncated prefixes of the agents' own tree paths.
- **`gt_name_mentions.txt` in recovery_literal_rep2 (21 files) is a substring match.** All
  165 matches are `soft_literal_listener`, a model an agent named itself in round 2. No
  other cell has the file.

## 4. `rsa_status.sh` at the end

```
sweep: /scratch/users/mcfrank/auto-psych/rsa_run1c   code: 12d4b205c1365706f2c8945557f7500b29b39720
task  cell                    state  stage   best model                             adm/rej/pruned  tokens  cost    agents  recovered  last exit
0     real_rep1               done   scored  surprisal_multimodal_l2_listener_2     30/4/31         265.2M  $47.82  35      -
1     recovery_literal_rep1   done   scored  softmax_belief_salience_listener       25/13/21        415.6M  $71.43  43      yes
2     recovery_salience_rep1  done   scored  base_literal_focal_lapse_listener      30/2/25         271.8M  $41.88  32      yes
3     real_rep2               done   scored  preemption_softmax_listener            30/4/34         215.5M  $39.97  35      -
4     recovery_literal_rep2   done   scored  valence_pragmatic_mixture_listener     29/6/25         361.1M  $58.95  36      yes
5     recovery_salience_rep2  done   scored  distractor_bayesian_salience_listener  30/7/25         333.9M  $53.11  39      yes
```

## 5. Changes for the driver session

Items 1-10 are infrastructure found while running; none affected run 1's results.
Items 11-14 bear on how to read them.

1. **`--qos=long` is invalid on this account** (verified 2026-10-07 with
   `sbatch --test-only`: "Invalid qos specification" on both `normal` and `mcfrank`).
   - Mike's association has only `high_p,normal`. The `long` QOS exists (7 days, <= 32 CPUs)
     but is not his.
   - `-p mcfrank` runs under QOS `owner` and accepts up to 7 days (a 6-day test request
     was accepted). `-p normal` refuses anything over 48 h.
   - Fix: `scripts/rsa/slurm/submit.sh` `qos_args` adds `--qos=long` for `TIME` > 48 h,
     which would be rejected. For > 48 h it should require (or switch to) `-p mcfrank`
     instead of adding a QOS.
   - Fix the same advice in:
     - `HANDOFF_sherlock_run1.md` (§0 Resources);
     - `CLAUDE.md` (Cluster & live runs: "`--qos=long` refuses limits under 48 h");
     - `scripts/outer_loop_live/README.md:204` and `run_live.sbatch:23`;
     - `scripts/subjective_randomness/slurm/README.md:86`;
     - `docs/onboarding/running_a_live_experiment.md:437` and `docs/onboarding/troubleshooting.md:52`;
     - `docs/auto_rsa/SMOKE_RESULTS_2.md:405`.

     Check whether any SR/live launcher also adds `--qos=long` automatically.
2. **Default partition.**
   - Every sbatch header and `submit.sh` default to `-p normal`, where fairshare is low.
   - The lab's owner node is `-p mcfrank` (24 cores, 192 GB): use it first, then
     `-p owners --requeue` (preemptible, so only for work that restarts cheaply).
   - Suggest `PARTITION=mcfrank` as the RSA default, sized to fit 24 cores: 6 x 4 CPUs
     worked, at 15-19 GB MaxRSS, so `MEM=24G` would also do.
   - The same applies to the 12 sbatch headers with `#SBATCH --partition=normal`
     (`git grep -n -- '--partition=normal' scripts`): the RSA setup and array, the live
     `run_live`/`setup`, and the SR holdout, impossible-holdout, retry, analysis and
     reanalyze jobs.
3. **Venv location.**
   - `_env.sh` (SR, sourced by RSA) puts the venv in `$WORK_ROOT/venv`, on `$SCRATCH`.
     Sherlock's rule 8 (`/etc/agents/AGENTS.md`) requires Python envs in `$GROUP_HOME`.
   - Make `$GROUP_HOME/venvs/<sweep>` the default, and have `rsa_status.sh` take the venv
     from `VENV_PY`/`UV_PROJECT_ENVIRONMENT` instead of hard-coding `$WORK_ROOT/venv`.
4. **`SSL_CERT_FILE`.** Export `SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt` in `_env.sh`
   when it exists. Without it, every HTTPS fetch from uv's Python fails on el7.
5. **`VENV_MODE=wheels` as the default for prepare** on Sherlock. The locked `uv sync`
   cannot succeed while uv.lock pins contourpy 1.3.3.
6. **opencode.**
   - Document the npm install (`$GROUP_HOME/software/npm-global`, nodejs/25.3.0) as the
     standard path, since the module is too old.
   - Consider defaulting `OPENCODE_BIN_DIR` to that location when it exists.
7. **prepare on a compute node.** `prepare_data.sh`'s header says to run it on a login
   node, which Sherlock's rules don't allow for a venv build plus data processing. Compute
   nodes reached every data host, so add a small `prepare.sbatch` (or document an
   `sbatch --wrap`).
8. **Handoff doc.**
   - §6's rsync commands hard-code `rsa_run1` and `login.sherlock.stanford.edu`. Use a
     `WORK_ROOT` variable and the `sherlock` SSH alias, which rides Mike's ControlMaster.
   - §4's cost should say $1.42 per agent run.
   - §5 should list the two el7 failures (SSL, bash 4.2).
9. **Review the two fixes made here:** 19ae7f2 (`rsa_setup.sbatch` comparison) and 12d4b20
   (`check_agent_tree.sh` guards plus test).
10. **Sherlock's `~/auto-psych` `main` has 10 commits not on GitHub** (latest "added final
    recovery results", a8839de). They were left in place, and the checkout is now on
    `auto-rsa`. Mike's to push or reconcile.
11. **The recovery verdict's held-out clause has no power on the salience data.**
    - `recovered = rmse <= 0.01 or gap >= -2*SE`. On the salience data, rsa_l1 and rsa_l2
      (RMSE 0.05) pass the held-out clause: 13.5-14.8 behind the ground truth, SE 8.9.
    - So "recovered" there rests on RMSE alone. On the literal data the seeds fail it by
      under 3 lpd.
    - Suggest:
      - drop the "or" clause, or require both criteria;
      - calibrate it by applying the verdict to the seeds, as §3 does, and record that
        baseline in `recovery.json`;
      - report the closest live model's distance as well as the exported one's (a live
        model was closer in all four cells).
12. **Selection and pruning reward fitting the training conditions, not generalising.**
    - The loop exports, and prunes at 2 x dse, on trial-level PSIS-LOO over the training
      conditions. In real_rep1 this pruned every one of the top six models on held-out
      conditions. It exported one 59-74 lpd worse than them, and no better than the seed.
    - Real cells also ended with 4 and 1 live models.
    - Options:
      - select and prune on grouped CV (leave-one-unit-out over training conditions,
        matching how the test set is held out);
      - a larger prune multiplier (the SR live series moved 2 -> 4 for a similar collapse);
      - a floor on live models.
    - real_rep1's SE of 71 on a -7.5 difference, against per-source gaps of 0.5-17, is
      also worth a look in `evaluate_heldout`.
13. **Agents' internet access: a decision for the team (Mike).** Agents can reach the web,
    and they used it:
    - code from the lab's `langcog/pragmods` repo (salience_rep2, 70+ requests);
    - Google Scholar and paper PDFs from OSF, escholarship and MIT Press (salience_rep1);
    - Hawkins et al. 2015 (literal_rep1).

    No agent fetched a source dataset, but nothing prevents it. In a real cell that would
    leak the held-out conditions, and in recovery cells literature on the ground truth
    weakens the claim. The options run from no network beyond the model API, through an
    allowlist (docs only), to recording and reporting only (the status quo). Until
    decided, keep reporting `agent_activity.md` per cell.
14. **`gt_name_mentions` scan: match whole identifiers.** It flagged an agent-coined
    `soft_literal_listener` as the ground truth `literal_listener` (165 matches, 21
    files). Use word boundaries (`\bliteral_listener\b`) so a real mention isn't buried.

### Sherlock reference

The consolidated operating notes for Sherlock live in a global Claude Code skill on
Mike's Mac, `~/.claude/skills/sherlock/` (`SKILL.md`, plus `references/stan.md`). It is
not in this repository, so a cloud session can't read it. What the changes above rely on:

- **Partitions:** `mcfrank` (owner node, never preempted, off fairshare, 7-day cap) >
  `owners` (preemptible, `--requeue`) > `normal` (48 h, low fairshare).
- **Storage:**
  - venvs and installed software in `$GROUP_HOME`;
  - job I/O on `$SCRATCH` (purged after 90 days; never refresh timestamps to avoid it);
  - per-job temp in `$L_SCRATCH_JOB`;
  - keep heavy I/O off the NFS `$HOME`/`$GROUP_HOME`.
- **el7 userland:**
  - glibc 2.17, so only manylinux2014 wheels install;
  - bash 4.2: guard empty arrays under `set -u` with `${a[@]+"${a[@]}"}`;
  - git 1.8: no `git -C`, no `branch --show-current`, no `worktree`;
  - uv's Python needs `SSL_CERT_FILE`.
- **Login nodes:** inspection and `sbatch` only. No loops, watchers or installs. Poll the
  scheduler at most once a minute.
- **Testing:** test scripts that will run on Sherlock under its bash, not only CI's bash 5.
