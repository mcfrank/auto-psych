# Run D2: 15 real people on each arm's smoke paradigm (2026-10-28/29)

Both studies filled and both jobs stopped `finished` after collecting (smoke tests are never
modelled). Work root `$SCRATCH/auto-psych/rsa_open_smoke`, code `fd20e669` (re-staged from
`6c532b05`: "Run D1 loose ends (git limit, 24-h window), PI 2026-10-29"), clean checkout
(`git_dirty: false` in both manifests). The PI approved the open paradigm (`review.json`, by
mcfrank, 04:30Z, sha256 `b0b54591…`) and the spend in chat before launch. Aggregates are in
`data/rsa_open/smoke/` (no `rows.jsonl`, `block.json`, `private/` or page files).

| | open (bet) | fixed (choice) |
|---|---|---|
| job | 50073267 | 50073272 |
| Prolific study | 6ae2d13d8059323f7c4caf6d | 6ae2ef7343bb24748f612f57 |
| recruiting | 21:53 to 00:00 PT (2 h 07) | 00:02 to 01:47 PT (1 h 45), after open's slot freed |
| trials | 24 test + 2 catch + 1 practice | 13 test + 2 catch + 1 practice |
| responses, kept | 15, 15 | 15, 11 |
| excluded | 0 | 4 (27%): missed a catch trial 3, comprehension on the second reading 2 (one person both) |
| comprehension, catch | 100%, 100% | 87%, 87% |
| same answer on nearly every trial, fast | 0, 0 | 0, 0 |
| minutes, median (p10, p90) | 7.9 (5.4, 14.6) | 2.9 (1.7, 6.9) |
| paid | $1.00 for 6 min | $0.50 for 3 min |
| effective rate at the median | $7.64/h | $10.39/h |
| Prolific cost | $20.00 | $10.00 |

Spend $30.00 (balance $78.67 → $48.67), as approved. Each study carried the blocklist filter
(`previous_studies_blocklist`, 11 earlier studies) besides country, language and approval rate.
No second submissions or people from another study.

## For the driver

1. **The open paradigm runs long.** Its pilot (simulated, `gemini-3.8-flash`) estimated 6.0 min;
   real people took a median of 7.9 (p90 14.6), so the pay works out to $7.64/h at the median
   against $10/h intended. The fixed pilot estimated 2.3 min against 2.9 real. Worth a look before
   stage 1 prices bet paradigms: `timing.json` now has a live point for bet (27 trials, 471 s median).
2. **The fixed arm excluded 27%, the open arm none.** Small samples (15 each), but the fixed
   paradigm's comprehension and catch rates (87%) are below the open one's (100%); both simulated
   pilots passed 100%, so the pilot does not predict these rates. One person failed both checks.
   If this holds at N, the fixed arm loses more of its sample to exclusions.
3. **The handoff's live gate conflicts with re-staging.** Setting `confirm_live_recruitment: true`
   in the checkout's `open_live.yaml` makes the checkout dirty, and `stage_code.sh` refuses to
   re-stage from a dirty checkout: the first D2 jobs (50072224, 50072229) stopped at staging, before
   any deploy. They were relaunched with the checkout clean and the confirmed config as a copy in
   `$WORK_ROOT/private/open_live_confirmed.yaml`, passed as `CONFIG=` (every reader takes
   `$CONFIG`); the copy was deleted afterwards. The handoff (and the campaign's launch, which also
   needs a clean checkout at the frozen tag) should say to do it that way.
4. **A test deploy's draft is never published.** Test mode creates a draft without the blocklist;
   live mode creates and publishes a new study with it. So D1's drafts (6ae246b9…, 6ae24887…)
   stay unpublished, and the PI's test rows in `/results` (no study id, or the other page's) can't
   match a live study.

## Left for the PI

- One fixed-study submission is AWAITING REVIEW in Prolific (the other 14 and all 15 open ones were
  approved by the study's completion setting); approving or rejecting it is the PI's.
- D1's two unpublished drafts can be deleted in the dashboard.
