# Reference-game data: what exists (2026-11-15)

Written by Claude Opus 5.5 from the repository's records; not reviewed by a person. "Trials" are modelled trials
(catch and practice trials excluded); "displays" are distinct displays a theory sees (objects, features, the word or
prior query, and the probe where the paradigm has one).

## Literature (block E of every open chain: 50,587 forced-choice trials)

| source | licence; in the repo | participants | trials | trials/person | displays |
|---|---|---|---|---|---|
| pragmods (Frank & Goodman; MTurk) | MIT; committed | 4,378 | 6,703 (forced choice of 10,168 rows) | 1 | 51 |
| Sikos et al. 2021 | CC-BY; committed | 4,482 | 4,482 | 1 | 79 |
| Mayn & Demberg 2026 | CC-BY; committed | 254 | 16,764 | 66 | 43 |
| Mayn & Demberg 2023 (4 experiments) | none stated; pin only, fetched on Sherlock | 228 | 15,048 | 66 | not in the repo |
| Mayn & Demberg 2022 (main + pilot) | none stated; pin only | 115 | 7,590 | 66 | not in the repo |
| **total** | | **9,457** | **50,587** | | 173 in the committed three |

## Closed campaign (October 2026; forced choice, $0.50 for 3 min, 13 designed choices + 2 catch per person)

Its live and pre-registered data are not used by the open campaign.

| | recruited | kept | trials | displays |
|---|---|---|---|---|
| pilot (chain 0's design at N = 20) | 20 | 20 | not modelled | |
| chain 0, experiments 1-3 | 200, 201, 201 | 185, 193, 185 | 2,405, 2,509, 2,405 | 52 each |
| chain 1, experiments 1-3 | 200, 201, 200 | 189, 165, 193 | 2,457, 2,145, 2,509 | 52 each |
| chain 2, experiments 1-3 | 196, 200, 200 | 178, 181, 190 | 2,314, 2,353, 2,470 | 52 each |
| **9 chain experiments** | **1,799** | **1,659** | **21,567** | 468 |
| pre-registered final experiment (OSF s48jv) | 1,225 | 1,108 | 14,404 | 312 (156 EIG + 156 random) |

(Chain 0's experiment 1 also modelled its catch trials at the time, `CAMPAIGN_LOG.md` #55; counted here as 13.)

## Open campaign

| | paradigm | recruited | kept | trials/person | trials | displays |
|---|---|---|---|---|---|---|
| run D2 smoke tests | rehearsal designs | 30 | | | never modelled | |
| **pilot 1** (open chain 1, first registration) exp 1 | bet allocation | 142 | 123 | 24 | 2,952 | 24 |
| exp 2 | choice + probed rating | 163 | 151 | 30 | 4,530 | 30 |
| exp 3 | begun, not finished | on Sherlock | | | | |
| **stage 1, open chain** exp 1 | choice (+ rating design) | 200 | 190 | 25 | 4,750 | 50 |
| exp 2 | two 7-point ratings | 200 | 157 | 25 | 3,925 | 50 |
| exp 3 | production | 250 | 228 | 30 | 6,840 | 60 |
| **stage 1, fixed chain** exp 1-3 | forced choice | 400 each | 354, 338, 354 | 13 | 4,602, 4,394, 4,602 | 52 each |
| **stage 1 total** | | **1,850** | **1,621** | | **29,113** | 314 |

Stage 1's people overlap across its two chains (registered: a person at most once per chain), e.g. 69% of open
experiment 2's sample had done a fixed-chain study.

## Totals by use

| | trials | of which forced choice |
|---|---|---|
| literature E | 50,587 | all |
| closed campaign (live + final) | 35,971 | all |
| open pilot 1 (two experiments) | 7,482 | none (bets; choice with ratings) |
| stage 1 | 29,113 | 13,598 fixed + 4,750 open |
