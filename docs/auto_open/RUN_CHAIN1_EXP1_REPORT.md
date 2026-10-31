# Open chain 1 (`chain_0`), experiment 1 (2026-10-29/31)

Registered campaign (https://osf.io/78t69, tag `rsa-open-prereg`, commit `5dc1e170`). Work root
`$SCRATCH/auto-psych/rsa_open_live`, staged once from the tag with a clean checkout (`git_dirty:
false` in the manifest). Live recruitment was confirmed in a copy of the config outside the
checkout (`$WORK_ROOT/private/open_live_confirmed.yaml`, passed as `CONFIG=`), as the handoff says.
Aggregates are in `data/rsa_open/live/open_chain_0/` (the list in the handoff's "Bring back",
except as noted at the end).

## Design and review

| | |
|---|---|
| test deploy | job 50182787, 2.8 h; page `https://auto-psych-2c5da-ro-open-c0.web.app/e1-open-c0/`, draft 6ae3e4f9598161f006b6c68f (never published) |
| paradigm | `pragmatic_reference_bet_allocation` (sha256 `d264b957…`), `submitted_by: designer`, first cycle |
| response | bet: 10 points over the pictures, after a one-word message or (prior queries) before one |
| materials | SVG, 3 domains (robots, houses, snowmen), scenes of 3 or 4 objects |
| trials | 24 test + 2 catch (+ 1 practice) |
| N, pay, minutes | 142, $1.40, 7.8 min; $265.07 with fees |
| EIG | 3.64 bits, 0.0137 bits per dollar (floor 0.0063) |
| pilot (`gemini-3.8-flash`) | comprehension 100%, catch 100%, readable 100%, 7.9 min estimated |
| PI review | approved by mcfrank, 2026-10-30T04:10Z (`review.json`) |

## Collection

| | |
|---|---|
| live job | 50237391 (job 1 of 4) |
| Prolific study | 6ae41a81d525be2a1c9901d6, published 21:17 PT, collection ended 23:17 PT (2 h 00) |
| blocklist | none, by design: the chain's first study (a person takes part once per chain, §4b) |
| responses, kept | 141 (142 recruited), 123 |
| excluded | 18 (12.8%): missed a catch trial 15, comprehension on the second reading 4 (one person both) |
| comprehension, catch | 97.2%, 90.4% |
| same answer on nearly every trial | 1 (p51: points 3,2,2,3 on 23 of 24; also missed a catch trial, so excluded) |
| fast | 0 |
| returning (earlier RSA studies) | 0 |
| minutes, median (p10, p90) | 9.1 (5.8, 19.1); kept people 8.8 (5.7, 16.9) |
| pay at the median | $9.27/h |
| enjoyment (1–5) | mean 3.89, n 141: 4 / 12 / 29 / 47 / 49 |
| Prolific cost | $265.07 |

## The loop

38.0 h, 23:17 PT on 2026-10-29 to 13:20 PT on 2026-10-31 (registered estimate about 20 h). Admission limit 73 min (2 × the median starting fit;
the starting fits took 16–161 min). Five rounds of six slots; **the leader changed in every
round**, so the stale-round stop never applied, and every leader after the first is discovered.

| after | leader | admitted, rejected | critique (significant of 8) |
|---|---|---|---|
| seeds | rsa_l2_singleton_feat_color_valence_l0 (seed) | 6 seeded | – |
| round 1 | lexical_competition_speaker_l2_2 | 5, 3 | 6 |
| round 2 | asymmetric_topic_prior_retention_l2 | 5, 2 | 3 |
| round 3 | oddball_literal_topic_retention_l2 | 6, 2 | 5 |
| round 4 | heterogeneous_listener_type_mixture | 4, 3 (1 no file) | 3 |
| round 5 | **speaker_alternative_necessity_tom** | 5, 1 (no file) | 4 |

Ledger (64 lines): 25 admitted; 12 rejected, of them 10 too slow to fit within 73 min (one
repair left the file unchanged) and 2 slots whose agent wrote no `candidate.py` even on retry;
27 pruned at the end (20 by margin on exp1, 2 × clustered SE; 7 ineligible: more than 4 SE behind
the best on the earlier corpus E).

**Exported live set** (`export.json`), grouped-CV ELPD behind the best, in nats (SE):

| model | source | exp1 | E (earlier corpus) |
|---|---|---|---|
| speaker_alternative_necessity_tom (best) | round 5, explore | 0 | 181 (58) |
| minimal_exemplar_diagnostic_mixture_l2 | round 5, refines round 4's leader | 3.0 (4.4) | 171 (57) |
| heterogeneous_listener_type_mixture | round 4, explore | 11.7 (7.5) | 185 (50) |
| rsa_l2_singleton_feat_color_valence_l0 | seed | 217 (49) | 0 |

Agent spend so far (chain): $109.52, of it candidates $104.51, critique $3.50, designer and pilot
about $1.51.

## For the driver

1. **The new data and the earlier corpus disagree.** The three discovered models beat the seed by
   about 217 nats on exp1 (4.4 SE) but are 171–185 nats behind it on E (3–3.6 SE), just inside the
   4-SE eligibility guard; a fourth (`diagnostic_competition_l0_retention_l2`, near the top on exp1)
   was pruned for crossing it. The carried set keeps both sides, so experiment 2's design is
   choosing between models that fit the bet paradigm and a seed that fits the earlier corpus.
2. **The loop took 38 h, not about 20.** The starting fits took about 5 h and each round about
   6.5 h; with a new leader every round, all five rounds ran. Job 1 paused experiment 2 with 7.9 h
   left (under the 8 h a design needs), and job 2 (50237409) started its design at 13:35 PT on
   2026-10-31. At about 43 h an experiment, three experiments take about 130 h of the four jobs'
   192 h.
3. **27% of candidate attempts were too slow** (10 of 37) against the
   73-min limit, which the slow seed `isolated_graded_costly_l3` (161 min) helped set.
4. **Pay ran under $10/h**: 9.1 min median against 7.8 paid ($9.27/h), as the pilot underestimated
   time again (by about 15%, less than D2's third). Prolific's $8 pause was never near.
5. **`newest_block_cv.json` is per-trial**: log-likelihoods for each of the 2,952 kept trials
   (123 people × 24) under each model, indexed by the campaign's person and trial numbers. It holds
   no responses and no Prolific ids, and the handoff lists it, so it is brought back; say if it
   should stay on Sherlock instead.

Not brought back: `lists.json` (27 MB, the per-person stimulus lists), `page_preview.html` (a
page file) and the pilot's simulated rows and transcripts (only `pilot/pilot.json`).
