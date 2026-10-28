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

**Not yet checked (the user's):** a test submission on each page (`?PROLIFIC_PID=test-<you>`; it
really submits) and each draft in Prolific's dashboard (name, description, pay, minutes, the
blocklist filter).

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
   node). Relaunched unchanged (49997977), and it passed. Not fixed: two live jobs starting together
   can hit it again (`_git` in `src/pipelines/outer_loop/deployment/manifest.py`); a longer limit
   would do.
