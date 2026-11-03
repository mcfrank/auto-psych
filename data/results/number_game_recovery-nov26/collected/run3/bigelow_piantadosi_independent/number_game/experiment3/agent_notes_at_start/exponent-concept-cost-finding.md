---
name: exponent-concept-cost-finding
description: Number-game data favour multiples-of-m over powers/squares/cubes; fitted extra prior cost on exponent concepts gained ~47 nats MAP over erlang_interval_simplicity_search
metadata:
  type: project
---
Round 3 (2026-10-30), cell_1/experiment1: adding a fitted extra log-prior cost on powers/squares/cubes to erlang_interval_simplicity_search raised MAP loglik from -873.7 to -826.3 (cost fit at about 5). Key items: 32,4,16→12 and 64,4→72 (people answer "multiples of 4"). Adding residue classes (x mod k = r) hurt by 20 nats (75,48,12,3 is ≡3 mod 9). Removing contains-digit rules hurt by 52. Submitted as `multiplicative_not_exponential_concepts`.

**Why:** saves re-running these concept-space screens.
**How to apply:** when MAP-screening, compile p_yes with `model.replace_rvs_by_values([model['p_yes']])[0]`. Otherwise compile_fn uses random RV draws and the per-trial predictions are garbage.
