---
name: minerva-echo-exemplar
description: exp3 MINERVA echo over category+interval traces (Jaccard^tau) smoke loglik -4283 vs best ELPD -3587
metadata:
  type: project
---
exp3 iter_0 cand_2 `minerva_category_trace_echo`: traces = 46 rules + all 5050 intervals, activation = Jaccard(examples, trace)^tau × rule strength, echo = p_yes, plus lapse/guess. Smoke loglik -4282.8 (best model ELPD -3586.7), so ~700 nats behind — exemplar family stays far behind ([[attention-exemplar-finding]], [[normalized-attention-similarity]]). Smoke fit 100+100 draws took ~90 s (dense U×K dot, K≈5100).
**Why:** exemplar brief slot recurs; no size principle costs hundreds of nats.
**How to apply:** don't expect exemplar variants to compete; keep them fast and simple.
