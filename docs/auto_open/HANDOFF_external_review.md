# A second opinion on what to do next: the open RSA discovery campaign (2026-11-18)

Written by Claude Opus 5.5 at the request of the PI (Michael C. Frank) for a reviewer from a different family of
AI models. The reviewer is not assumed to have the repository or any earlier conversation, so this document gives
the full context. Numbers are from the campaign's records as of 2026-11-18. Where this document gives an opinion,
it says so; please feel free to disagree with it.

---

## 0. What we are asking you

We have run an automated discovery loop twice, in two campaigns. In the loop, AI agents propose cognitive theories
as code, design experiments, collect data from real people, and let cross-validated model comparison decide among
the theories. The second, "open" campaign let an agent change the experimental paradigm from one experiment to the
next.

The results so far raise four questions. We would like your view on each:

1. **Complexity.** How should theories be penalized for complexity in a principled way?
   - Counting fitted parameters failed: the agents moved the complexity into code structure instead.
   - We now have code-size measures (§6). Should they be shown to the theorist agents, used in selection, both, or
     neither?
   - If they are used in selection, where does the exchange rate between nats of fit and units of code come from?
2. **Varying the paradigm across experiments.** Changing the response format from one experiment to the next seems
   to have swamped everything else (§5.2, §5.4). Theories ended up right or wrong for format-specific reasons.
   - Is that the right diagnosis?
   - If so, what design keeps the open arm's flexibility without that confound?
3. **Two concrete proposals from the PI:**
   - (a) every experiment collects several dependent variables (choice, ratings, bets, production) rather than one;
   - (b) a one-time base dataset that collects every dependent variable on displays already used, given to every
     chain from the start.
   What are their strengths, risks and better alternatives? More generally, how could the open arm's flexibility be
   used *within* an experiment rather than *across* experiments?
4. **Anything we are missing.** Wrong inferences from the evidence below, analyses that would settle a question
   cheaply with the data we already have, or design flaws.

A useful answer gives:
- your diagnosis;
- a recommended design for the next registered study, with its main risks;
- one to three analyses we could run on existing data before committing.

Please be concrete; we can act on specifics.

---

## 1. The project

**auto-psych** is an automated cognitive-science pipeline. Each step of the loop:
- coding agents write theories as self-contained Python files (here in `memo`, a probabilistic programming language
  compiled to JAX), each fitted with NUTS (numpyro);
- a design step chooses the next experiment's stimuli to maximize expected information gain (EIG) among the current
  theories;
- the experiment runs on Prolific with real participants;
- theories are compared by grouped cross-validation (whole displays or conditions held out per fold);
- new theories are proposed in rounds, guided by a critic agent's posterior-predictive checks.

**The domain is reference games** (Frank & Goodman, 2012, the "pragmods" paradigm):
- **The display.** A participant sees two to four objects, each with binary features (e.g. a face with or without
  glasses and a hat).
- **The utterance.** A speaker uses one word, naming one feature (e.g. "glasses").
- **The task.** In the standard forced-choice version, the participant picks the object the speaker meant.
- **Prior trials.** These ask "which object is the speaker talking about?" with no word heard, which measures the
  prior over objects.

**Rational speech act (RSA) models:**
- **The literal listener** (L0) picks uniformly among the objects the word is true of.
- **The pragmatic speaker** (S1) picks a word in proportion to exp(α · log L0), so rationality α sets how strongly it
  prefers informative words.
- **The pragmatic listener** (L1) inverts S1 under a prior over objects; L2 and L3 repeat the recursion.

The literature adds salience priors (which objects people expect to be talked about), graded semantics, utterance
costs, noise, and alternatives.

**Response formats ("paradigms") used so far**, each scored by a likelihood "link" with its own nuisance parameters:

| format | what people do | link |
|---|---|---|
| choice | pick the referent | categorical over classes of identical objects; temperature and lapse |
| rating | rate on a 1-7 scale how likely a probed object is the referent | ordered probit on the logit of the tempered probability; centre, spread, lapse |
| bets | split 100 points across the objects | Dirichlet-multinomial with concentration κ on the tempered, lapse-mixed posterior |
| production | choose a word to describe a target | the theory's speaker, tempered, with a lapse |
| two measures | e.g. a choice then a rating of a probed object on the same screen | sum of the two parts |

A theory may define its own speaker. If it doesn't, the speaker is derived from its listener:
S(u | r) ∝ L(r | u) over the scene's words.

---

## 2. The closed campaign (October 2026)

- **Design:** 3 chains × 3 live experiments, all forced choice. Each experiment had about 200 people and 52 displays,
  chosen by EIG from a pool of 1,597 plain displays; each person saw 13 of them.
- **Loop:** each experiment ran rounds of candidate theories until two rounds brought no change. At most 8 theories
  were carried from one experiment to the next.
- **The bar:** theories had to beat 5 starting models (literal listener, RSA L1, L2 and two salience variants) and 12
  "promoted seeds" from earlier runs on the published data.
- **The pre-registered test** (OSF https://osf.io/s48jv; 1,225 recruited, 1,108 kept, 14,404 trials). Against the
  best promoted seed:
  - chain 0's final: +197.9 ± 35.1 nats (z 5.6);
  - chain 2's final: +60.8 ± 22.9 (z 2.7);
  - chain 1's final: +8.1 ± 24.1 (z 0.3).

  All three beat the best starting model (z 8-10).
- **Afterwards, a consolidation:**
  - a superset model whose 13 mechanisms (from the three finals) switch on and off;
  - backward elimination by grouped CV left 7 mechanisms and 9 parameters at depth 1;
  - on the (by then spent) pre-registered data, that lean model was 3.0 SE behind chain 0's final.

The closed finals had 9-17 parameters, and some set numbers by hand inside the code.

---

## 3. The open campaign

### 3.1 Question and design

**The question** (Yarkoni, 2022, "the generalizability crisis"): do discovery loops that **design their own
paradigms** produce theories that generalize across operationalizations better than loops that refine theories on
**one fixed paradigm**?

**Two arms run the same harness and differ only in who chooses each experiment:**
- **Open:** an agent designer chooses the paradigm (response format, instructions, trial structure) and the
  displays. It works within a budget, uses an EIG tool, and pilots the design with simulated participants.
- **Fixed:** the closed campaign's forced-choice paradigm, with displays chosen by the same EIG tool.

**The registered plan** (OSF https://osf.io/wurh5, revision 2, 2026-11-13):
- 4 chains per arm × 3 experiments each.
- Each experiment's loop runs 3 rounds of 6 theorist agents: 3 explore, 2 refine the incumbent, 1 refines a theory
  of its choice.
- A feasibility gate after stage 1 (one chain per arm).
- A blind "bakeoff" paradigm at the end.
- $266.67 of participant budget per experiment.
- Agents are Gemini 4 models; sessions are 60 minutes, sandboxed, with no network.

### 3.2 What theories are, and how they are chosen

- **Theory format.** A theory is a *paradigm-free* file: `PARAMS` (numpyro priors) plus
  `listener(params, ctx) -> P(object)` for one display, and optionally `speaker(params, ctx)`. The links in §1 turn
  it into a likelihood for each format.
- **Joint fitting.** A theory is fitted jointly to every block of data the chain has: the published literature plus
  each live experiment. Its own parameters are shared across blocks; each block has its own link parameters.
- **Admission gates:**
  - an import allowlist and no network;
  - listener and speaker contracts (valid distributions, finite gradients);
  - convergence (≤ 0.1% divergent transitions, R-hat ≤ 1.05, bulk ESS ≥ 100);
  - a 4-hour fit limit;
  - **no hand-set numbers** (from revision 2): any numeric literal outside the priors other than 0, 1, a named
    tolerance and a named mask value is refused.
- **Selection:**
  - grouped 5-fold CV within every block;
  - a guard: a theory may not fall more than 4 clustered SEs behind the best on any block;
  - **the one-SE rule** (from revision 2): of the theories within 1 clustered SE of the best on the newest paradigm,
    the one with the **fewest fitted parameters** leads.
- **Theorists' brief.** The theorists are told that short theories with meaningful parameters are the aim.
- **The critic** writes test statistics, which are scored against posterior-predictive replicates and passed to the
  next round's theorists.

### 3.3 What has run

| run | what | people kept | trials |
|---|---|---|---|
| open pilot 1 (first registration) | exp 1: bet allocation (24 displays); exp 2: choice + probed rating (30); exp 3 begun, not finished | 123, 151 | 2,952, 4,530 |
| stage 1, open chain | exp 1: choice (50 displays); exp 2: two 7-point ratings of probed objects (50); exp 3: production (60) | 190, 157, 228 | 4,750, 3,925, 6,840 |
| stage 1, fixed chain | 3 forced-choice experiments, 52 displays each | 354, 338, 354 | 13,598 |
| closed campaign | 9 chain experiments + the pre-registered final | 1,659 + 1,108 | 21,567 + 14,404 |
| published literature | pragmods, Sikos et al. 2021, Mayn & Demberg 2022/2023/2026 | 9,457 | 50,587 |
| **total** | 23 studies | | **123,153** |

Stage 1's two chains drew partly on the same participants: 69% of open experiment 2's sample had taken part in a
fixed-chain study.

---

## 4. What happened in the open campaign

### 4.1 Open pilot 1

- **Hand-set numbers.** The experiment-1 theories carried a median of 9 hand-tuned constants (up to 27) beside 2-5
  fitted parameters. The exported theories set 17, 24 and 27 numbers by hand. This led to the no-hand-set-numbers
  rule.
- **Slow fits.** Fits took up to 2.5 hours and 35 GB, and the chain ran 8 days for 3 experiments. This was fixed:
  fits are now about 10× faster with the same density.

### 4.2 Stage 1 (2026-11-13 to 11-15): the gate failed on one item

- **Every item passed except one:** design, data quality, loop health and time (≤ 48 h per experiment; actual 8-14
  h).
- **The failure.** Open experiment 2 (two 7-point ratings) did not give the carried incumbent more than 4 clustered
  SEs over a chance theory: it got 1.2.
  - No theory, carried or new, got more than 2.3 SEs from chance on that block.
  - Three of the 50 displays carried 75% of every theory's gain.
  - The design tool had predicted this paradigm to be the most informative of the three (3.97 bits).
- **The registered consequence:** stage 1 becomes a pilot, and a revision is registered anew before any further
  chains.

### 4.3 What the two arms' theories looked like (stage 1)

Of every distinct theory admitted (58 open, 59 fixed):

| | open arm | fixed arm |
|---|---|---|
| salience prior | 97% | 97% |
| depth-2 recursion | 83% | 85% |
| singleton objects in the prior | 72% | 76% |
| graded truth by feature count | 57% | 0% |
| its own speaker model | 64% | 3% |
| feature-tier case indicators ("intermediate tier", "cloned minimal tier") | 0% | 80% |
| median lines | 71 | 136 |
| surviving hypotheses that name a critic statistic | 0 of 14 | 9 of 18 |

- **The incumbents had 1-2 parameters in nearly every case**, so the one-SE rule kept parameter counts low.
- **Complexity moved into structure instead.** The fixed arm's incumbents grew from 110 to 181 and 205 lines over
  three experiments. Its final (`fixed3`, 1 parameter) is a case table over display configurations. It is no longer
  RSA.
- **The open arm's incumbents changed with each new paradigm:**
  - open 1 (choice): graded semantics with noise on literally false objects;
  - open 2: repaired that noise after it collapsed on the ratings;
  - open 3: added a production speaker, a cost-weighted choice mixed with a fallback, after the production
    experiment.

### 4.4 Cross refit

Every stage-1 theory was refitted jointly on all seven stage-1 blocks.
- Each arm's final is best on its own arm's paradigms.
- The other arm's final is more than 2 SEs behind there.
- The fixed arm's theories fail on production: all are 477-595 nats behind plain RSA there.
- The open arm's final is 4-5 SE behind `fixed3` on the forced-choice blocks.
- On the dual-rating block nothing discriminates.

---

## 5. The standard-RSA grid and every theory on all 123,153 trials

### 5.1 The grid

**36 textbook RSA theories**, crossed on four axes:
- **depth:** 0 (literal listener) to 3;
- **where a salience prior enters:** none, the top listener only, or shared by every level;
- **what is salient:** an object's feature count, or being a singleton (no identical object in the scene);
- **semantics:** Boolean, or graded (a word is true to 1 / the object's feature count).

That gives 0-2 parameters per theory.
- **A cost axis was dropped.** A word's extension share, with its own weight, was a second knob on the speaker's
  preference for specific words. It traded off exactly with recursion depth, and every utterance is one word, so
  there is no production cost to model.
- **Fitting and scoring.** Every grid theory, plus 31 seed and discovered theories, was fitted on all 23 studies at
  once, with grouped 5-fold CV inside every study block. Scores are out-of-fold log predictive density in nats over
  plain RSA (depth 1, nothing else).
- **Convergence flags.** 24 of the 36 grid fits and 11 of the 27 scored others are flagged as not converged. A
  diagnostic re-fit of three of them traced every failing fold to one stuck chain on one nuisance link, the second
  rating of open pilot experiment 2. The theories' own parameters converged. Scores are given with and without that
  study; no conclusion here depends on it.

### 5.2 What the grid says

On all the data:

| contrast (mean over grid pairs that differ only on it) | effect, nats | pairs where it helps |
|---|---|---|
| depth 1 over 0 | +2,229 | 4/6 |
| depth 2 over 1 | −59 (converged pairs: level) | 3/10 |
| depth 3 over 2 | −84 (converged pairs: level) | 5/10 |
| singleton prior over none (top listener) | +2,139 | 6/6 |
| feature-count prior over none | +420 | 12/12 |
| prior shared across levels vs top listener only | −97 (converged pair: 0) | 5/12 |
| graded over Boolean, depth 1 | −103 | 0/5 |
| graded over Boolean, depth 0 | +4,237 | 3/3 |

- **The best standard theory is the simplest with a salience prior:** depth 1, the top listener's prior favouring
  singletons, Boolean meanings, 2 parameters. It scores +2,318 over plain RSA.
- **The literature alone tells a slightly different story.** There, graded meanings helped a depth-1 listener (+41)
  and depth 2 edged out depth 1 (+21). Graded meanings and a second level did much the same work. Neither survives on
  all the data.

### 5.3 The seed and discovered theories against the best grid theory

On all the data. "Hand" means numbers set by hand; complexity is measured on the active code (§6).

| theory | origin | vs best grid | parameters (+ hand) | lines | Halstead volume |
|---|---|---|---|---|---|
| best grid theory (d1, singleton, Boolean) | textbook | 0 | 2 | 32 | 2,371 |
| closed consolidation superset | closed, post hoc | +1,505 | 15 + 11 | 118 | 13,036 |
| `contrastive_extremity_noisy_speaker` | closed chain 1, exp 2 | +1,481 | 9 + 1 | 65 | 4,727 |
| `contrastive_anchor_decision` (fixed 2) | stage 1 fixed | +1,474 | 2 | 181 | 11,257 |
| consolidation, lean | closed, post hoc | +1,327 | 8 + 8 | 103 | 9,680 |
| `prototype_tier_confusion` (fixed 1) | stage 1 fixed | +1,079 | 2 | 110 | 7,441 |
| `softmax_belief_listener` | closed chain 1, exp 1 | +970 | 7 | 62 | 4,659 |
| `sole_exclusion_shadowing_heuristic` (fixed 3) | stage 1 fixed | +906 | 1 | 205 | 13,021 |
| `attentional_buffer_anchor_sampling` | open pilot 1, exp 2 | +735 | 2 + 123 | 377 | 26,556 |
| `rsa_l2_singleton_feat_color_valence_l0` | closed seed | +657 | 6 | 67 | 4,647 |
| `speaker_alternative_necessity_tom` | open pilot 1, exp 1 | +63 | 2 + 27 | 103 | 9,284 |
| `l1_graded_semantics` (open 2) | stage 1 open | −531 | 2 | 69 | 5,350 |
| `complexity_gated_retrieval_decision` (open 3) | stage 1 open | −544 | 2 | 101 | 8,171 |
| `extension_noisy_graded_semantics` (open 1) | stage 1 open | −1,654 | 2 | 68 | 5,218 |
| closed chain 2's three theories | closed chain 2 | about −12,800 | 13-14 | | |
| not scored: closed chain 0's three incumbents and chain 1's third | timed out at 48 h | | 12-17 | | |

**Where the gains come from:**
- **The closed and fixed theories gain mostly on forced-choice data,** much of it data their makers never saw: the
  closed pre-registered final (+186 to +336 over the best grid theory) and stage 1's fixed experiments.
- **The fixed arm's final (`fixed3`) is the best theory on 14 of the 24 study columns,** 11 of them unseen by its
  makers. But it loses on stage 1's open experiments (−768) and on the pilot's bets (−389).
- **The open arm's theories win on the format they were last selected on and lose on others:**
  - open 1 loses 885 nats on bets and 519 on ratings, because its graded semantics give literally false objects real
    probability;
  - open 3 is the best theory of all on production (+584 over the best grid theory) but loses its own chain's rating
    experiment (−488) and the pilot's bets (−767).
- **Closed chain 2's theories collapse on bets** (about −14,000 nats on that one study) and nowhere else. Most
  likely they give some objects nearly zero probability, which a forced choice tolerates and a bet does not.
- **Production.** Most theories' speaker is the listener reversed, so it inherits the listener's salience prior. A
  literal listener's derived speaker beats plain RSA by 347 nats on production. The singleton prior that wins on
  every choice study costs on production.

### 5.4 Choice dominates the pooled data

About 109,000 of the 123,000 trials are choices. Ratings, production and bets together are about 14,000.

---

## 6. Complexity measures (built 2026-11-16 and 17)

**The measures** are computed on a theory's *active code*:
- comments, docstrings and dead code (unused helpers, assignments, imports) are removed;
- for the consolidation's variant files, switched-off mechanisms are removed by partial evaluation, and a test
  checks the pruned file predicts exactly as the original.

| measure | definition |
|---|---|
| lines | lines of the active code, in the author's layout |
| AST nodes | nodes of the parsed active code |
| Halstead volume | (N1 + N2) log2(n1 + n2) over operator and operand tokens (N total, n distinct) |
| compressed bytes | zlib size of the active code; a crude description length |
| parameters | the theory's fitted parameters |
| hand-set numbers | as counted by the no-hand-set-numbers gate; numbers inside priors are exempt |

**Validation on the grid,** whose complexity is known by construction:
- AST nodes, Halstead volume and compressed bytes rise every time an axis is switched on (161 of 161 pairs, counted
  on the earlier 66-theory grid);
- lines miss 33 of those pairs, so lines is the weakest measure.

**The frontier.** The project's results page plots fit against each measure. The grid's frontier, the best textbook
score at each size, is the reference: a theory above it fits better than any textbook theory as small.
- `contrastive_extremity_noisy_speaker` is on the frontier: no theory with as little code fits better.
- The fixed arm's case tables are high but far to the right, at 3-5× the code.

**A calculation that matters for question 1.** Take the most literal "principled" penalty, two-part MDL at one bit
per bit of compressed code. Then `contrastive_extremity_noisy_speaker`'s extra 339 bytes over the best grid theory
cost about 1,880 nats, which is more than its 1,481-nat gain. Every discovered theory would lose to the 2-parameter
textbook theory.
- Compressed Python is a poor code for theories: syntax and boilerplate are redundant.
- So the exchange rate cannot come from file size alone.
- An alternative is to estimate it from transfer: how much of a theory's fit on data its makers saw fails to carry to
  data they did not. The data for that exist: every score is marked seen or unseen per study.

---

## 7. Observations

These are the drafting agent's reading, not settled.

1. **The parameter rule worked on what it measured and not on what matters.** Theories stayed at 1-2 parameters
   while their code tripled.
   - The no-hand-set-numbers rule did catch something real: the pilot's best theory carries 123 hand-set numbers.
2. **Cross-validation guards against overfitting parameters, not against overfitting structure.** A theorist who
   sees all the data and writes a case table for its displays has fitted structure; grouped CV cannot catch that.
   The campaigns' only true out-of-sample tests are prospective scores on data no theorist saw. These exist for both
   campaigns, partly.
3. **Changing the response format every experiment confounds three things:**
   - the new format;
   - the new displays;
   - a new likelihood link with its own nuisance parameters.

   A theory can win or lose on any of them. In the open arm, theory changes tracked the newest format: each
   incumbent repaired the last format and broke an earlier one. No accumulation of mechanism is visible. Whether
   the designed *displays* mattered cannot be told from these data.
4. **One format can make or break a theory for reasons unrelated to reference.**
   - Bets punish any probability on literally false objects.
   - Ratings depend on a link (an ordered probit with cutpoints) that may be misspecified. The one rating-only
     experiment carried almost no signal.
   - Production scores a speaker that most theories never wrote.
5. **The best textbook model is a strong bar.** Depth-1 RSA with a singleton prior gets most of the reachable gain on
   choice data. The bar for discovery should be it, not plain RSA.
6. **The fixed arm's case tables transfer better than expected** to other forced-choice data (the closed campaign's
   final and experiments). They do not transfer across formats.

---

## 8. The PI's current thinking (2026-11-18)

- Rules against parameters are not the way forward. Use a code-complexity measure such as Halstead volume to
  penalize theories, but do it in a principled way. Open question: do we simply give the theorists the score?
- Varying the paradigm across experiments may not be doing what we want. The response format is so important that
  it swamps other changes and makes theories right or wrong for the wrong reasons.
- It is not clear that we learned anything from the differences in *displays* across the open chain's experiments:
  if they mattered, other differences swamped them.
- Ideas under consideration:
  - require every paradigm to collect multiple dependent variables throughout;
  - or collect a base dataset with every dependent variable on displays already used, so every theory is adapted to
    every format from the start.
- Generally: find a way to use the open arm's flexibility *within* experiments rather than across them.

---

## 9. One view on the options

This section is the drafting agent's own view, offered for you to critique.

**On complexity:**
- **Show the measure to the theorists, and select on it, but don't make it a hard penalty yet.**
  - Replace parameter count in the one-SE rule with an active-code measure. AST nodes is the hardest to game;
    Halstead volume is close.
  - Show every theorist its theory's value, the incumbent's, and the frontier.
- **Fix the exchange rate before looking at the scores it will judge.**
  - A rate estimated from the existing seen/unseen split (how many nats of seen-data gain fail to transfer per unit
    of code) is the most defensible.
  - The literal MDL rate above is too harsh for Python and would reject everything.
- **Keep the no-hand-set-numbers rule,** or count hand-set numbers in the measure.
- **Watch for gaming:** code golf, or moving structure into data tables. Both still show up in AST nodes.

**On the paradigm:**
- **Fix the measurement, and let the open arm vary what is manipulated.**
  - A standard multi-measure protocol for every experiment: for example, choice on every trial, plus a rating or a
    bet on a subset, and a production block. Its links are validated once on a base dataset.
  - The designer chooses the displays and the *manipulations*, which the closed campaign never varied:
    - prior manipulations (familiarization, valence, colour), which the pragmods literature shows are informative;
    - speaker knowledge;
    - alternatives available to the speaker;
    - instructions;
    - within-experiment conditions.
  - Then a difference between experiments is a difference in what was manipulated, measured the same way.
- **A base dataset is cheap insurance.** Collect every measure on about 50-100 displays already in the corpus, once.
  It gives every chain all formats from round 1 and calibrates the links before any theorist sees them.
- **Score every experiment's theories prospectively** on the next experiment's people, as now, and report that as
  the main check on structural overfitting.

**Analyses we could run on existing data first:**
1. **How much does code size explain transfer?** Regress each theory's gain on unseen studies on its gain on seen
   studies and its code size.
2. **Are the links the problem?** Refit the grid with alternative rating and bet links and see whether the format
   effects shrink.
3. **Does a speaker fix production?** Give the best grid theory an explicit speaker (with its own α and no prior)
   and see whether production stops penalizing the salience prior.

---

## 10. Constraints

- **Money:** about $267 of participant budget per experiment, and about $100-130 of agent spend per chain.
- **Infrastructure:** fits run on a university cluster (Stanford's Sherlock). An experiment takes 8-14 hours of
  loop time plus recruitment.
- **Registration:** every new design is registered anew on OSF before data. Data already seen cannot be used to
  confirm a choice made after seeing it. The closed campaign's pre-registered data are spent for confirmatory claims.
- **Agents** are Gemini 4 models in a sandbox with no network, and never see held-out data. People's data are
  de-identified, and no free text is shared.
- **The open arm's designer** must produce a paradigm that passes data-quality checks: comprehension ≥ 0.8, catch
  trials ≥ 0.8, exclusions ≤ 25%.

## 11. Pointers

All files are in https://github.com/mcfrank/auto-psych, branch `auto-psych-open`. You do not need them to answer.

| file | what |
|---|---|
| `docs/auto_open/PREREG_V2.md` | the registered design |
| `docs/auto_open/STAGE1_REPORT.md`, `STAGE1_ANALYSIS.md` | the stage-1 gate and what stage 1 learned |
| `docs/auto_open/CROSS_REFIT_REPORT.md` | stage 1's theories on all stage-1 blocks |
| `docs/auto_open/RSA_GRID.md`, `RSA_GRID_REPORT.md` | the grid, its results, complexity |
| `data/rsa_open/grid/all_studies/` | the final report and results page |
| `src/rsa_open/complexity.py` | the complexity measures |
| `docs/auto_rsa/CAMPAIGN_LOG.md`, `CONSOLIDATION.md` | the closed campaign |
