# OSF registration of the final experiment

The plan is `PREREG_FINAL.md`. This file is how it is frozen on OSF: the steps, and the answers
for OSF's **"OSF Preregistration"** template, section by section. Each answer is short and points
to the plan, which is linked in full.

**OSF no longer takes uploaded files** (its platform change, 2026-10). So the registration links a
**tagged commit** of the public repository: `rsa-prereg-final`, https://github.com/mcfrank/auto-psych/tree/rsa-prereg-final.
- The registration text quotes the tag's **full commit SHA**. A tag can be moved or deleted, but a
  commit SHA is fixed by the content of every file in that commit, so the link stays verifiable.
- It also quotes the key files' sha256 (the "Other" answer), so the fingerprints are on OSF itself.

**Why OSF:** the git freeze commit fixes the code. The OSF registration adds a public, timestamped
record that nobody on the project can edit or withdraw (PI 2026-10-15; `CAMPAIGN_LOG.md` #62).

## Steps

1. **The freeze** (driver):
   - fill `PREREG_FINAL.md` §8 and `scripts/rsa/live/rsa_prereg.yaml`;
   - commit the design files to `data/rsa/prereg/`;
   - push.
2. **The tag** (driver):
   - check the freeze: `uv run python -m src.rsa.prereg.bundle --out <scratch dir>` verifies every
     hash against the config and the plan, and refuses an unfrozen plan (its zip is not needed);
   - tag the commit that holds the freeze and these instructions, `rsa-prereg-final` (annotated), and push the
     tag;
   - fill in its full commit SHA in the "Other" answer below (a copy for pasting; the tagged file
     cannot hold its own commit's SHA).
3. **The registration** (PI, on osf.io: the account, and a registration cannot be withdrawn):
   1. Create a project, "auto-psych: RSA final experiment" (or a component of an existing
      auto-psych project).
   2. Choose **Registrations → New registration → "OSF Preregistration"**, and paste the answers
      below.
   3. **Make it public** (PI 2026-10-15), not embargoed. It must be submitted, and approved,
      before the study is published on Prolific.
   4. Submit, then approve the registration from the confirmation email. **OSF makes it final
      only after that approval, or after 48 hours.**
4. **Record it** (driver): put the registration's URL, its approval date and the tag's commit SHA in
   `PREREG_FINAL.md` as Addendum A, in the campaign log, and as `osf_registration` in
   `rsa_prereg.yaml`.
   - **The live launch (handoff step 7) waits for this.**
   - The test deploy (step 6) does not: it publishes nothing.

## Answers for the template

### Study information

**Title:** Do agent-discovered models of reference resolution predict new people better than
the models they started from? A pre-registered prospective test.

**Authors:** Michael C. Frank (and collaborators, as on the paper).

**Description:**
- **The campaign:** coding agents in an automated loop (auto-psych) proposed computational models
  of how people interpret a speaker's word in simple reference games. Three independent chains
  each ran three live experiments of 200 people. Each ended with a final model:
  - `boundary_avoidable_ambiguity_listener`;
  - `feature_surprisal_ensemble_listener`;
  - `competition_endpoint_binding_lapse_l2`.
- **This study:** it tests, on new participants and with predictions frozen in advance, whether
  each final predicts choices better than the strongest of the models the campaign started from.
  Those are the 12 promoted seeds, the best models of an earlier phase fitted to published data.
- **The record:** the full plan, frozen predictions, design and analysis code are in the tagged
  commit `rsa-prereg-final` of the public repository (https://github.com/mcfrank/auto-psych/tree/rsa-prereg-final): the plan is
  `docs/auto_rsa/PREREG_FINAL.md`, and the hashes are under "Other".

**Hypotheses:**
- **H1_f, for each final model f:** on new participants, f's summed log predictive density
  exceeds that of the best promoted seed.
- **The bar:** the promoted seed with the highest log predictive density on the new data, chosen
  after the fact.
- **The tests:** three one-sided tests, with Holm correction at familywise α = 0.05.
- There are no secondary hypotheses (`PREREG_FINAL.md` §2).

### Design plan

**Study type:** Experiment, with a within-participant design of display types. Nothing is
manipulated between participants. The question is which models predict the choices.

**Blinding:** No blinding is involved. Participants do not know the hypotheses. The analysis is a
fixed script applied to frozen predictions, with no analyst choices.

**Study design:**
- **The task:** each participant sees 13 designed displays of 2–4 objects with features (for
  example, people with hats or glasses). They either hear one word ("glasses") or no word (a
  "mumble" trial), and click the object they think the speaker means.
- **Trials:** there are also 2 catch trials and a practice trial.
- **The displays:** G in all, from a pool of 1,597:
  - half chosen by the campaign's expected-information-gain design rule, to tell the finals and
    the 17 bar models apart;
  - half drawn at random;
  - both halves under fixed quotas of display types.
- Details: `PREREG_FINAL.md` §4.

**Randomization:**
- Each participant gets one of N trial lists, assigned in arrival order: balanced subsets of the
  design, so every display gets the same number of lists.
- Trial order, object positions and the pictured domain are randomized within each list, by a
  fixed seed (2026).

### Sampling plan

**Existing data:** **Registration prior to creation of data.**
- The models were developed and fitted on earlier data: published studies plus the campaign's 9
  studies, 70,841 trials.
- No data from this study exist at registration.
- Its participants exclude everyone in any earlier study of this task.

**Explanation of existing data:** The earlier data were used only to fit the models before their
predictions were frozen. The predictions files are attached, with their hashes.

**Data collection procedures:**
- **Recruitment:** Prolific, desktop only, approval rate ≥ 98%.
- **Exclusions at recruitment:** anyone in any earlier study of this task, by Prolific's
  blocklist.
- **Pay:** $0.50 for a study listed at 3 minutes.
- **Consent:** the consent text approved under the existing IRB protocol (the subjective-
  randomness protocol).

**Sample size:** **N participants**, from §8 of the plan, filled at the freeze.

**Sample size rationale:** set by power analysis (`PREREG_FINAL.md` §7, `power.json`,
`design_power.json`). It is the smallest number of displays and participants at which:
- a simulation calibrated to the campaign's weakest comparable effect gives ≥ 0.90 power for
  every final;
- the empirical scaling of the campaign's own tests gives ≥ 0.90 power.

**Stopping rule:**
- Collection ends at N submissions, or after 48 hours. A study short at 48 hours is paused and
  its data analysed as they are, with the shortfall reported.
- There are no interim analyses and no extension based on results.

### Variables

**Manipulated variables:** none between participants. Display type (number of objects, word or
mumble trial) varies within participants, by design.

**Measured variables:** the object chosen on each trial.

**Indices:**
- Each model's log predictive density: the sum over modelled trials of the log of its frozen
  probability for the chosen object class. Identical objects form one class.
- Differences between models (`PREREG_FINAL.md` §6.1).

### Analysis plan

**Statistical models:**
1. For each final f: diff_f = lpd(f) − lpd(best promoted seed).
2. Its standard error is clustered by display: sqrt(G · var over displays of each display's
   summed difference).
3. z_f = diff_f / SE_f, with one-sided p_f = 1 − Φ(z_f).
4. The script is `src/rsa/prereg/analysis.py`, attached and tested on simulated data. Nothing is
   fitted to the new data.

**Transformations:** none. Catch trials are removed from the modelled data.

**Inference criteria:** Holm's step-down procedure over the three p-values at familywise α =
0.05. H1_f is supported when Holm rejects its null.

**Data exclusion:**
- A participant who misses any catch trial is excluded.
- There are no other exclusions: none by response time, and no outlier removal.

**Missing data:**
- Only submitted, complete sessions are analysed.
- Prolific returns and timeouts never reach the data.

**Exploratory analysis** (reported, not tested):
- each final against the best starting model and against each of the 17 bar models;
- every model's log predictive density;
- the primary comparison within each half of the design.

### Other

**Deviations:** any deviation is reported with its reason, in a dated addendum to
`PREREG_FINAL.md` and in the paper.

**The frozen materials:** the git tag `rsa-prereg-final` of https://github.com/mcfrank/auto-psych, commit
`<full SHA, filled in at registration>`.

| file (in that commit) | sha256 |
|---|---|
| `data/rsa/prereg/design/design.json` (design sha256, `Design.sha256`) | `bc7d60e5b84f0e09b1578da54e2ae73975b67e8f05f97a10cea75df6f8ceee02` |
| `data/rsa/prereg/design/trial_lists.json` | `a11b6cc6aeac1b70b8ba47174099bc1e264db722cea25ce44d17dfbb54a12672` |
| `data/rsa/live/finals_pool_preds_final.json` (the finals' frozen predictions) | `6a0a476f24e21166dd5af9d3f6538addd39db011b62d96a1adcd505a1d94f4bf` |
| `data/rsa/live/bar_pool_preds.json` (the bar's frozen predictions) | `704575376be1305a92506f7270c27c67524f26ffeb0391687d4178d5974de269` |
| `src/rsa/prereg/analysis.py` (the analysis) | `981bb8c0a01de84265ba6491a20a0768bc25bb7a79ff3b901ee5479c97ef910a` |
| the pooled data the models were fitted to (not public: trial-level) | `19af0f544ed5c0e0cf902fdce55c259ad2fbaf697ccef6a21f9617375f114f23` |
