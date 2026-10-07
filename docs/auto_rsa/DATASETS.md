# Candidate datasets beyond pragmods (survey, 2026-10-06)

## Status (2026-10-07): ingested

Built by `src/rsa/ingest/` (`uv run python -m src.rsa.ingest.run --sources ...`;
combine with pragmods via `src.rsa.ingest.combine`). Details, exclusions and
gaps: `src/pipelines/outer_loop/projects/rsa_reference/data/README.md`.

| source | data | licence | in repo | participants (included) | paper's N reproduced |
|---|---|---|---|---|---|
| `mayn_demberg_2026` (PLoS One) | OSF 5ab3f | CC-BY 4.0 | CSV + pin | 306 (254) | yes: 300/8/292/23/15/254 |
| `mayn_demberg_2023` (Open Mind) | GitHub refgame_stimuli_methods @ d7d4aec | none | pin only (CSV in `data/rsa/external/`) | 237 (57/55/56/60) | yes (Table 1) |
| `mayn_demberg_2022` (CogSci) | GitHub refgame_cogsci22 @ b3a2b5e | none | pin only | 115 (47 pilot + 68 main) | main 68 (cited in the 2026 paper); paper not retrievable |
| `sikos_2021` (PLoS One) | supporting files S1-S3 | CC-BY 4.0 | CSV + pin | 7,488 (5,625) | yes, every N and exclusion count |

Not ingested: **Duff, Mayn & Demberg 2026** (OSF 7uwx9/ad685): participants got
feedback after every reference-game trial and a speaker pre-training, so the
choices are learned under reinforcement. Excluded by the PI: Franke & Degen
2016, child data, imagined-child / ChatGPT-speaker conditions (f5nmv,
perceptions_of_chatgpt), slider studies (erbn3), the listener-adaptation study
(5d2f6), Franke, Tsvilodub & Carcassi 2024.

The survey below is unchanged from 2026-10-06.

Survey of public reference-game data that could join the seed data. Sources:
problang-v2's and the pragmods paper's bibliographies, plus web search.
Fit score: **A** = maps to our trial schema (objects x binary features,
word heard, forced choice) with modest work; **B** = needs a schema
extension; **C** = not usable.

Files checked in a cloud session: GitHub-hosted data only. PLoS, OSF,
Zenodo and escholarship are blocked there, so those sets are unverified.

## A: maps to our schema

| Dataset | Paradigm | Size | Where | Verified |
|---|---|---|---|---|
| **Franke & Degen 2016**, PLoS One | Simple/complex implicature + fillers; 3 objects on 2 feature dimensions, some features not nameable; fixed set of 4 messages | ~60/experiment, **66 listener trials per person**; comprehension + production | PLoS supporting files (S1/S2 Data CSV): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154854 | no |
| **Mayn & Demberg series** (CogSci 2022; Open Mind 2023, 2025; PLoS One 2026; Duff et al. 2026) | Franke & Degen game on Prolific, with individual differences (reasoning, ToM, working memory); 2025 manipulates speaker identity (adult vs 4-year-old) | ~60+ per study, many trials each, with covariates | OSF: https://osf.io/5ab3f/ (2026 PLoS), https://osf.io/7uwx9/ (Duff 2026), https://osf.io/w7yqg?view_only=fa7f9034042946f0928d2c772e0a23ad (2025), prereg https://osf.io/dh8wm (2023; data link in the article) | no |
| **Sikos, Venhuizen, Drenhaus & Crocker 2021**, PLoS One | Close Frank & Goodman 2012 replication over more context types; listener, speaker, salience conditions | trial-level in supporting files | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248388 | no |
| **Yoon & Frank 2019**, JECP (`ejyoon/simpimp_rs`) | Children 2-5, tablet; 2 objects as sets of items, one-noun utterance | 2,792 trials, 180 children, 8-16 trials each | GitHub (no licence file; Zenodo 1285235) | **yes** |
| **Qing & Franke 2015** | Frank & Goodman replication, colour/shape | aggregate counts only (listener 2x3, production 3x4, salience prior) | problang app-04 | yes (counts) |

Mapping notes:
- **Franke & Degen and Mayn & Demberg:** a feature dimension's value becomes
  a 0/1 column. We would need to mark which columns have a word (some
  features cannot be named), a small schema addition. Theirs and the pragmods
  sequences are the only sets with many trials per person, which
  individual-level models need.
- **Yoon & Frank:** the display has to be rebuilt from `trial_type` x
  `item_num`, and the 2vs2 reading should be checked against `experiment/`
  stimuli. `sample2.csv` has positions and words; `sample1.csv` does not. The
  final file is already filtered (`log.csv` has the decisions). These are
  children, so treat them as a separate population.

## B: needs a schema extension
- **Degen et al. 2020** (`thegricean/overinformativeness`): production of
  redundant size/colour; free text coded to mentions. The final data are
  unconfirmed in the repo.
- **Degen & Franke 2012; Degen, Franke & Jäger 2013:** costly artificial
  messages. Data not located.
- **Kreiss & Degen 2020; Waldon & Degen 2021:** colour-adjective production.
  Data not located.
- **Monroe et al. 2017** (colors in context, GitHub): continuous HSL colours
  and free text, 57,946 messages.
- **Bohn et al. 2021/22** (`manuelbohn/mcc`): word learning with combined
  cues. The display is not stored per trial.

## C: not usable here
- Tangrams/convention formation (Hawkins et al.).
- Scalar-implicature star ratings (Peloquin & Frank). `rrrsa`'s pragmods data
  are aggregated pragmods.
- Politeness; embedded scalars.

## Duplicates
- **Vogel et al. 2014** is pragmods E8 (levels), already in the seed.

## To fetch or allowlist
- **Allowlist** `osf.io`, `api.osf.io`, `files.osf.io` and `journals.plos.org`
  in the cloud environment, or download into
  `src/pipelines/outer_loop/projects/rsa_reference/references/data/` from a
  local session.
- **Franke, Tsvilodub & Carcassi 2024** (https://arxiv.org/abs/2406.09012)
  names an OSF repository of human reference-game data collected to test LLMs.
- **Ask the authors:** Stiller, Goodman & Frank 2015; Fortier et al. 2023
  (Shipibo-Konibo children); trial-level Qing & Franke 2015; raw Frank &
  Goodman 2012 bets.
