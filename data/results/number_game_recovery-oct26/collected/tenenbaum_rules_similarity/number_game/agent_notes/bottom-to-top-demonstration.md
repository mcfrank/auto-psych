---
name: bottom-to-top-demonstration
description: exp3 iter_2 c1: compressed_rule_extension + first-shown min (n>=2) marks range bottom with own bottom_odds; 4-way range split
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_1 (`bottom_to_top_demonstration`) targets the critique below_asym_min_first (q=0.10). It is a copy of the incumbent with ranges split 4 ways (bottom-at-min x top-at-max) and a second generative credit, log1p(bottom_odds*share/q), for sets of 2 or more that start on their min. Raw data are sparse: in min-not-first, max-last sets, yes within 10 below the min is 0.80 (n=25), vs 0.57 (n=75) in min-first sets. The check passed: smoke LL -3093.1 (the incumbent gets -3094.5), 270 s per run (the incumbent takes 225 s).
**Why:** It is the only order cue left untried that the critique flagged. **How to apply:** check its LOO and novelty outcome. Related: [[compressed-rule-extension]], [[last-example-marks-range-edge]]
