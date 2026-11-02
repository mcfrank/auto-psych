---
name: final-digit-salience-finding
description: Fitted log-prior bonus on "ends in d" rules gained ~38 nats MAP over erlang_interval_simplicity_search; many other tweaks gained ~0
metadata:
  type: project
---
Round 3 repair (2026-10-30), cell_1/experiment1: MAP screens on erlang_interval_simplicity_search (base loglik -873.7).
- Gained nothing (<0.1 nat): fitted stray-noise rate, separate size-principle weight, fitted Erlang shape, a decade-aligned interval bonus, a bonus on the full 1-100 range, a multiples-of-m bonus (which was rejected as a near-duplicate).
- Big gains: "ends in d" bonus alone gave -835.4 (bonus about 2.2, cost about 4.0). Separate ends and contains bonuses gave -826.9. One digit-rule bonus gave -852.3.
- The largest per-set misfits: 92,82,27,42 (it over-uses "contains 2"), 62,32,22 and 83,38,58,68 (it over-uses "even"), and the square/power sets (see [[exponent-concept-cost-finding]]).
Submitted as `final_digit_salient_concept_search`.

**Why:** saves re-screening dead-end tweaks.
**How to apply:** for this base model, change which rules are in the hypothesis space or how they are priced. Tweaks to the likelihood, noise or interval prior do not move the fit.
