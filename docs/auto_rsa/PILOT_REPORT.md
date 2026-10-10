# Live pilot (stage 2): report

From the local Sherlock session, 2026-10-10, for the driver session and the PI.
`HANDOFF_live.md` §3, approved by the PI.

## 1. What ran

| | |
|---|---|
| job | 47283766: COMPLETED, 1 h 21 min (design ~45 min, then deploy, recruitment and collection) |
| code | 58e7048 (page, collection, functions and deploy code unchanged since stage 1's 90a06be) |
| work root | `$SCRATCH/auto-psych/rsa_live_pilot` (fresh: `rsa_live` was staged from 90a06be) |
| run | cell `live_rsa_c0_pilot`; page `https://auto-psych-2c5da-rsa-c0.web.app/e1-c0pilot/` |
| Prolific study | `6acabe1541f1407647c27606`: 20 places, $0.80, desktop, country / language / approval-rate filters |

**The config:** a copy of `scripts/rsa/live/rsa_live.yaml` in the work root, with only
`confirm_live_recruitment: true`, passed as `CONFIG`. The checkout stayed clean and the flag
was never committed.

**Earlier-participants blocklist:** none on the study, which is correct. The scope is
`project` (RSA only), and no RSA study had been published (only the two test drafts).

**The design** (chain 0, experiment 1, for N=20):
- 40 EIG picks: 2×3 word 1; 2×4 prior 3 / word 2; 3×3 word 1; 3×4 prior 4 / word 5; 4×4
  prior 9 / word 15;
- power 0.745 ± 0.010 (free 0.744), joint EIG 2.88 bits;
- 5 responses a display.

## 2. What the handoff asks

**Median completion time:** **119 s (2.0 min)**, against the 4-min estimate the pay
assumes. The 20 who completed took between 80 and 670 s:
- under 2 min: 10 people (80-111 s);
- 2-3 min: 6 people (127-182 s);
- 4.5 min: 2 people (265-266 s);
- 11 min: 1 person (670 s).

**Catch-trial exclusions:** **none.** `data/participants.json`: `n_included` 20,
`excluded` 0 (`max_catch_errors` 0), `n_responses` 20, `without_trials` 0, `other_study` 0.

**Each designed display's response count:** **exactly 5 for all 40**, every participant
with 12 rows (10 designed + 2 catch; the catch condition holds 40 rows).

**Complaints and returns:**
- no returns and no messages seen through the API;
- **1 timed out**: started, never finished and was not paid. Prolific refilled the place;
  that person has no data;
- the dashboard's messages are not readable through the API, so the PI should glance there.

## 3. Half the submissions are held for review: Prolific's speed threshold

| Prolific state | n | completion code | `time_taken_under_auto_approval_threshold` | times (s) |
|---|---|---|---|---|
| APPROVED | 10 | entered | False | 127-670 |
| **AWAITING REVIEW** | **10** | entered | **True** | 80-111 |
| TIMED-OUT | 1 | none | — | — |

- Every completer came back through the redirect with the code.
- Prolific auto-approves only submissions slower than its threshold (relative to the
  study's time estimate), and **holds faster ones for manual review**. The 10 fastest were
  held.
- Their data are complete and they passed both catch trials, so they look genuine. **The
  PI should approve them in the dashboard.** Until then the study stays `AWAITING REVIEW`
  and they are not paid.

## 4. What this means for the campaign (decisions for the PI)

1. **Pay:** at a median of 2 min, $0.80 is about **$24/h**, double the $12/h target. The
   options:
   - keep $0.80 (generous, which may help quality and speed);
   - or price to the measured time: e.g. a 2.5-min estimate at $0.50, about $12/h at the
     median. That brings the campaign from about $1,915 to roughly $1,200 (9 × 200 × $0.50
     plus the fee).
2. **The estimate drives the review hold.** With a 4-min estimate, every completion under
   about 2 min was held. Across the campaign's 1,800 people that would be about half held
   for manual review, which needs someone to approve them before they are paid.
   - A lower `estimated_completion_time` (e.g. 2-2.5 min) should hold far fewer.
   - Either way, someone needs to clear held submissions as the campaign runs.
3. **The welcome screen** says "about 3 minutes"; the measured median is 2. Align it with
   whatever estimate is chosen.
4. **Recruitment is fast:** 20 places filled in about 30-40 min (from 15:3x to 16:0x on a
   Saturday afternoon US time). 200 a study should take a few hours, inside the 3-hour
   poll in most cases.

## 5. Left in place

- **Pilot data stay on Sherlock:**
  - responses and participants, under the agent tree `66d0e02aca8b920a`'s
    `_runs/outer/experiment1/`;
  - raw responses with Prolific ids, in the cell's `private/`.

  Nothing was copied off Sherlock.
- **The Prolific study** awaits the PI's review of the 10 held submissions.
- **The test draft** `6acaaa90665e4a53bd804a51` (stage 1) and the older
  `6aca6c2a3e69a3f93c402401` are still unpublished drafts.
