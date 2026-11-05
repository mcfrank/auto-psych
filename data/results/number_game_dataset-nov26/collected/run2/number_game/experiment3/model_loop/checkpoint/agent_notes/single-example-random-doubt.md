---
name: single-example-random-doubt
description: exp3 n=1 flat-response mixture on span_interval_similarity: +37 MAP; converges only with floor/judge-share Dirichlet + log-normal band/span widths
metadata:
  type: project
---

exp3: on one-example sets, a share of trials answered at a flat rate (mixture), otherwise span_interval_similarity. +37 MAP, smoke -4682.4.

The first version (doubt x random_yes Betas, HalfNormal band_width/span_width) failed admission with R-hat 1.07. The fix (2026-11-02, repair_1) has two parts:
- parameterise the mixture as Dirichlet([yes_floor, no_floor, judge_share]), which gives well-identified masses;
- the real culprit: band_width and span_width were bimodal and swapped roles near 0, because doubt absorbs the n=1 trials and leaves the band unidentified. LogNormal(log .25, .5) and LogNormal(log .05, .5) fixed it: R-hat <= 1.01, ESS > 400 on 2x500.

**Why:** any graft that soaks up the n=1 trials leaves the incumbent's band width weakly identified.
**How to apply:** when a model makes n=1 trials flat, put log-normal priors on the widths. Related: [[range-band-single-leniency]] and [[size-discounted-anything-concept]].
