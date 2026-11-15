# What stage 1 learned (2026-11-15)

Written by Claude Opus 5.5 from the stage-1 run trees brought back (`data/rsa_open/live_v2/`) and refits run in the
cloud container; not reviewed by a person. Exploratory: stage 1 is a pilot (its gate failed, `STAGE1_REPORT.md`), and
everything here rests on one chain per arm.

## Bottom line

1. **The loop finds theories that beat plain RSA clearly, everywhere it looks**: by 6.6-6.9 SE on each chain's first
   experiment, and by 3-5 SE on the literature in a refit. The gains are real.
2. **But they do not yet beat the best starting theory on data the loop did not design**: on the committed literature
   the 6-parameter seed (`rsa_l2_singleton_feat_color_valence_l0`, from the closed campaign's starting set) ties the
   fixed arm's best and beats the open arm's by 2.4-3.6 SE. On each chain's own new experiments the discovered theories
   beat that seed by 5-6 SE. So much of what the loop learns fits the experiments it designed.
3. **The two arms learned in different ways.** Both settled on the same core, depth-2 RSA with a salience prior that
   favours singleton objects. Beyond it, the **open arm** built general mechanisms its paradigms demanded: graded truth
   (a word is less true of an object with more features) and explicit speaker models once production data arrived. The
   **fixed arm** built case tables over display types (80% of its theories use feature-tier indicators; its final,
   with one parameter, is 205 lines of indicators), and half its surviving hypotheses name the critic's statistics.
4. **The parsimony rules worked on what they measure, not on what matters.** No hand-set numbers, few parameters (2 in
   nearly every incumbent), but complexity moved into structure: the fixed arm's theories grew from 110 to 205-255
   lines over three experiments, and the one-SE rule counted a 205-line case table as the simplest theory.
5. **The gate failed on a paradigm, not a theory.** On the open arm's dual-rating experiment no theory got beyond 2.3 SE
   of chance; three of 50 displays carried 75% of every theory's gain.

So: on the right track, with a clear problem. The loop finds the right core and adds genuine mechanism when the
paradigm forces it (the open arm), but given one paradigm it overfits structure to that paradigm's displays (the
fixed arm). That is the open-versus-fixed hypothesis showing up in the pilot, at n = 1 per arm.

## 1. What the theories look like

Every distinct theory either chain admitted (58 open, 59 fixed), by what its code contains:

| | open | fixed |
|---|---|---|
| salience prior (`softmax_prior`) | 97% | 97% |
| depth-2 recursion | 83% | 85% |
| singleton objects in the prior or score | 72% | 76% |
| graded truth by feature count | **57%** | **0%** |
| a word's extension size (noise, cost) | 66% | 36% |
| its own speaker model | **64%** | **3%** |
| feature-tier case indicators (min/max/intermediate tiers) | **0%** | **80%** |
| lines (median, range) | 71 (32-105) | 136 (32-255) |
| surviving hypotheses that name a critic statistic | 0 of 14 | 9 of 18 |

The incumbents, in order:

- **open 1** `extension_noisy_graded_semantics_l2` (2 parameters, 70 lines): depth-2 RSA, salience prior, graded truth,
  semantic noise growing with the square of a word's extension.
- **open 2** `l1_graded_semantics_l2` (2, 70): the same with graded noise only in the middle speaker, so literally false
  objects get zero probability again; it repaired open 1's collapse on ratings (below).
- **open 3** `complexity_gated_retrieval_decision_l2` (2, 102): the same listener, plus a production speaker that mixes a
  cost-weighted pragmatic choice with a feature-availability fallback. Recognisably RSA, with hand-built functional
  forms (noise ∝ extension², cost = share × (2 + share − 1/features)).
- **fixed 1** `prototype_tier_confusion_l2` (2, 110), **fixed 2** `contrastive_anchor_decision_l2` (2, 181), **fixed 3**
  `sole_exclusion_shadowing_heuristic` (1, 205): increasingly elaborate case analyses of display configurations
  ("incomplete upper-tier literal referents", "cloned minimal tier", "sole excluded singleton"), the last one not RSA
  at all.

## 2. How well they do

**On their own experiments** (out-of-fold, as the loop scored them; clustered SE in parentheses):

| | incumbent over plain RSA | over the best seed | over the best carried theory | over its runner-up |
|---|---|---|---|---|
| open 1 (choice) | +289 (6.9 SE) | +171 (5.9) | | +22 (1.2) |
| open 2 (dual rating) | | −9 (−0.8) | +780 (2.4) | −9 |
| open 3 (production) | | | +696 (5.0) | +1 (0.6) |
| fixed 1 (choice) | +410 (6.6) | +89 (4.9) | | +0.3 (0.1) |
| fixed 2 | | | +36 (3.3) | +0.6 (0.2) |
| fixed 3 | | | +95 (4.2) | +6 (0.9) |

Within an experiment the top ten or so theories are tied; the real gains are over RSA, the seeds and what was carried
in. New paradigms moved the open arm a lot (+700-800 nats on ratings and production over the carried theories); the
fixed arm's later gains are small (+36, +95).

**Across a chain's paradigms** (each final incumbent's distance from the best theory on every block, at its
experiment's end; the guard allows 4 SE): the open arm's winners trade off across paradigms (open 2 is 3.7 SE behind on
E and 4.0 on its choice block; open 3 is 2.3 SE behind on the rating block); the fixed arm's are best or nearly so on its
repeated paradigm (fixed 3 pays 2.4 SE on E and 3.8 on its first block). No theory is best everywhere in the open arm,
which is what changing the paradigm is for.

**On the literature, refitted here** (the three committed sources, 27,949 trials, 164 units; grouped 5-fold CV, 4 chains
× 500 + 500; all converged). The full literature block (50,587 trials, including these) was every chain's first block,
so this is fit to data the theorists saw, not transfer:

| theory | over plain RSA | over the 6-parameter seed | Mayn & Demberg 2026 | pragmods | Sikos 2021 |
|---|---|---|---|---|---|
| fixed 2 | +633 (4.8 SE) | +39 (0.8) | −29 (−1.0) | +262 (4.1) | +400 (4.1) |
| seed, 6 parameters | +594 (4.3) | 0 | −7 (−0.6) | +171 (3.4) | +430 (3.8) |
| fixed 3 | +571 (3.9) | −23 (−0.4) | −100 (−2.2) | +256 (3.8) | +415 (3.9) |
| fixed 1 | +461 (3.7) | −132 (−2.3) | −63 (−2.7) | +171 (2.4) | +353 (4.1) |
| open 1 | +437 (4.0) | −157 (−2.4) | −15 (−0.5) | +112 (3.5) | +341 (3.9) |
| open 3 | +434 (3.9) | −160 (−2.6) | −17 (−0.8) | +98 (3.1) | +353 (3.9) |
| open 2 | +398 (3.1) | −196 (−3.6) | −67 (−3.0) | +69 (2.1) | +395 (3.8) |
| closed depth-4 reference | +387 (3.0) | −207 (−2.5) | −43 (−3.2) | +44 (0.8) | +386 (3.9) |
| competitor-confusion seed | +289 (3.9) | −305 (−3.6) | +2 (0.5) | +38 (1.2) | +248 (4.5) |
| RSA depth 2 | +39 (4.0) | −555 (−4.2) | +7 | +14 | +17 |
| literal listener | −864 (−4.5) | −1,458 (−6.9) | −429 | −512 | +77 |

(Source columns: over plain RSA.) The gains come from pragmods and Sikos; on Mayn & Demberg 2026, the largest source,
nothing beats plain RSA.

## 3. The failed item: open experiment 2

Two 7-point ratings per display (200 people, 157 kept). Every theory's gain over chance comes mostly from three of the 50
displays (320, 293 and 176 nats of the incumbent's 1,041); many displays give nothing or less. Two displays with the same
objects and different probes gave +293 and −49. The paradigm's predicted information was the highest of the three open
designs (3.97 bits), so the design tool's prediction and what the people's ratings carried diverged; that is claim 3's
measure, and worth a look before rating designs are allowed again. Experiment 1's winner collapsed here (261 nats over
chance against about 1,040 for nearly every other theory) because it gave literally false objects positive probability,
which the next theory repaired.

## 4. What it suggests for the revision

These are proposals for the PI, not decisions.

1. **Measure complexity structurally.** Parameters alone let structure grow unchecked. Candidates for the one-SE rule's
   tie-break and for reporting: AST nodes or lines of the theory's own code, the number of scene-case indicators, or a
   description length. The fixed arm's final would no longer count as the simplest.
2. **Keep theorists from patching the critic's statistics.** Half the fixed arm's surviving hypotheses name one. The
   critique could report what is wrong in words (which kinds of scene, which direction) without named statistics, or
   the brief could forbid hypotheses that cite them.
3. **The fixed arm reuses one display pool**, so a case table over its display types pays off; the open arm's changing
   paradigms do not reward that. This is the hypothesis, but it also means the fixed arm's theories may be the weaker
   baseline for the reason the study wants to test. Worth stating in the revision.
4. **The information item** asked the first incumbent to clear 4 SE; on the rating data nothing came near, so the item
   judged the paradigm. Options: keep it (it caught a weak paradigm), require it of the experiment's best theory rather
   than its first incumbent, or add a predicted-information check against realized information at design time.
5. **Cross-chain overlap** (69% of open 2's people had done the other chain's study) will dominate the registered
   sensitivity analysis; staggering the arms' studies, or a campaign-wide blocklist, would avoid it.

## 5. Next

The decisive test is whether each arm's theories hold up on the other arm's data. `HANDOFF_cross_refit.md` has Sherlock
refit the 13 theories above on all seven blocks at once (E plus both chains' six experiments), a dry run of the
bakeoff's fit A. Then the revision can be decided with that table in hand.
