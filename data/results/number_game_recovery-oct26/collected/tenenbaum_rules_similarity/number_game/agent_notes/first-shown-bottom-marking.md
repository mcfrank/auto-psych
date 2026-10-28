---
name: first-shown-bottom-marking
description: exp3 iter_2 c4: incumbent + bottom credit when a multi-example set opens on its min (bottom_odds); targets below_asym_min_first critique; check outcome
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_4 (refinement slot): `sequence_bracketed_ranges` is compressed_rule_extension plus generative bottom credit log1p(bottom_odds*[n>1 and s[0]==min]/draw_prob). The plain-range evidence is split 4 ways (bottom x top). It targets the critique below_asym_min_first: observed 0.094 vs the model's 0.171.
**Why:** order-only edge tweaks have failed novelty before ([[last-shown-top-discounted-ranges]]). **How to apply:** check whether it passed novelty before reusing it. Related: [[first-shown-top-marking]], [[exp3-iter2-critique-stats]]
