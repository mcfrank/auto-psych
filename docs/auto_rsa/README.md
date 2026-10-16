# auto-rsa docs

The RSA reference-game domain (`src/rsa/`), developed on branch `auto-rsa`, 2026-10-06 to 10-17.

**Read first**
- `CAMPAIGN_LOG.md`: every decision with its reason, how the domain differs from the
  subjective-randomness loop (§3), and every run in order with its headline (§4).
- `LOOP_ALGORITHM.md`: the live loop step by step, with every model set and scoring rule.
- `PLAN.md`: the running plan and decision tables, phase by phase.

**Results**
- Claim 1, existing data: `SHERLOCK_RUN1_RESULTS.md`, `SHERLOCK_RUN2_RESULTS.md`.
- Claim 2, the live campaign: `CAMPAIGN_C0_E1_REPORT.md`, `CAMPAIGN_C0_REPORT.md`,
  `CAMPAIGN_C1_REPORT.md`, `CAMPAIGN_C2_REPORT.md` (the last also compares the chains).
- The pre-registered test: `PREREG_FINAL.md` (the plan, frozen; addenda dated),
  `OSF_PREREG.md` (the registration, OSF s48jv), `PREREG_RESULT_REPORT.md` (the result).
- After the test: `CONSOLIDATION.md` (exploratory; the pre-registered data are spent).
- Figures: `figures/` (`figures/README.md` says what each shows and its data).

**Data:** `DATASETS.md` (sources and licences). Run records: `data/rsa/README.md`.

**Historical records.** These are kept as written; where one disagrees with a later doc, the
later one and `CAMPAIGN_LOG.md` hold.
- Handoffs to the Sherlock session: `HANDOFF_*.md`.
- Test runs: `SMOKE_RESULTS.md`, `SMOKE_RESULTS_2.md`, `REHEARSAL_REVIEW.md`,
  `REHEARSAL2_REPORT.md`, `REHEARSAL3_REPORT.md`.
- Live stages and incidents: `STAGE1_FINDINGS.md`, `PILOT_REPORT.md`, `SAFE_BROWSING_C2.md`.
- `phase_report_template.html`: the template of `scripts/rsa/build_phase_report.py`.
