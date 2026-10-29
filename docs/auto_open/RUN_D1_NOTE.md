# Run D1: test deploys of run C′'s designs (2026-10-28)

No money: each job put up its page and a Prolific draft and stopped with a STOP record. Work root
`$SCRATCH/auto-psych/rsa_open_smoke`, code `6c532b05` (staged fresh, so it includes run C′'s fixes,
32b493e8; the "re-stage before D2" step is already done).

| arm | job | page | Prolific draft |
|---|---|---|---|
| open (bet design) | 49995636, 15 min | https://auto-psych-2c5da-ro-open-c0s.web.app/e1-open-c0-smoke/ | 6ae246b913c7bfe61f451c26 |
| fixed | 49997977, 5 min | https://auto-psych-2c5da-ro-fixed-c0s.web.app/e1-fixed-c0-smoke/ | 6ae24887ed0aaf1248afe72c |

**Page sizes** (fetched, nothing submitted): each page is 29 kB and fetches one list per
participant. Open: about 360 kB a list (27 trials, pictures as inline SVG). Fixed: about 40 kB
a list (16 trials: 1 practice, 13 test, 2 catch), against run C's 360 MB fixed-arm page.

**Test submissions (the PI, 2026-10-28 evening).** One on each page; both stored in full:
- open, 21:14: list 1, 32 rows (1 practice, 24 test, 2 catch, 5 other screens). The link carried
  only `PROLIFIC_PID`, so the row has no study id (the page takes `STUDY_ID` from the URL) and the
  monitor, which keeps one study's rows, counted 0.
- fixed, 21:19: list 0, 21 rows (1 practice, 13 test, 2 catch, 5 other screens), with
  `STUDY_ID` and `SESSION_ID` recorded from the URL. The link carried the open study's id, so this
  monitor counted 0 too. `/results` scopes rows by page: neither page's fetch returned the other's row.
- With each row given its own study id in memory, the monitor's per-person code reads both: fixed
  kept (comprehension and both catch trials passed, 1.2 min); open excluded for a missed catch
  trial (trial 24, "hat", 10 points on the glasses-and-scarf robot after 2.2 s; the PI was
  clicking through, and the display order there was the identity, so not a page fault), 3.3 min.

**The drafts** (read through the API): both unpublished, "What do people mean?", the neutral
description with the paradigm's minutes; open $1.00 for 6 min, fixed $0.50 for 3 min, 15 places
each, $20.00 and $10.00 with fees. Their filters are country, language and approval rate only: a
test draft carries no blocklist (`create_draft_study` adds it only in live mode, which creates a
new study), so D2 publishes new studies, not these drafts. Balance $78.67.

## Three failures on the way

1. **A fresh work root always failed** (49980054, 49980055, 1-2 min): `open_live.sbatch` made
   `$CELL_DIR/private` before `stage_code.sh`, which refuses a work root with cells but no
   `code_commit`. Fixed in 6c532b05 (the cell is made after staging; a test checks the order). The
   campaign's root, `rsa_open_live`, would have failed the same way.
2. **The launcher defaults to `normal`**, where a 24-CPU/96-GB job waited hours; moved to
   `owners,normal`, and 32b493e8 sizes smoke jobs at 4 CPUs/16 GB. `PARTITION=mcfrank,owners,normal`
   is worth passing too (the lab node had room).
3. **The provenance record's `git status --porcelain` timed out** (10 s) in the fixed job
   (49995640): the checkout is on `$HOME` (NFS), git 1.8 stats every file, the node's cache was cold
   and the open job ran the same command at the same moment (5.7 s cold, 0.7 s warm on a login
   node). Relaunched unchanged (49997977), and it passed. The driver raised the limit to 120 s
   (e6716ee7).
