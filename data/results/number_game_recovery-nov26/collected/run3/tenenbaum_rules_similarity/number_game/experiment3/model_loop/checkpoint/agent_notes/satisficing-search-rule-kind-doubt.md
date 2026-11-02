---
name: satisficing-search-rule-kind-doubt
description: exp3 iter_3 c5: range_first_satisficing_search + incumbent's rule-family doubt criterion; smoke OK ll −3652.9, 19 s
metadata:
  type: project
---
exp3 iter_3/candidate_5 (2026-10-31), satisficing_search_rule_kind_doubt. Built by a python patch of models/range_first_satisficing_search.py, swapping in the criterion block from rule_family_doubt_criterion.py. Smoke check OK: ll −3652.9, 19 s sampling. Novelty was not checked. The risk is a same-round duplicate if the incumbent slots (c3/c4) graft satisficing onto the incumbent.
**How to apply:** if it ties, satisficing (effort≈0.1) and rule-kind doubt stack. See [[range-first-satisficing-search]], [[rule-family-doubt-criterion]].
