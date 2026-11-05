---
name: span-fallback-rule-violation
description: exp3 iter4 c0: no-rule multi sets take "range of examples" as concept, outside-span shares c_viol; +19 MAP, 0 new params; smoke -4017.0
metadata:
  type: project
---
exp3 iter_4 candidate_0 (2026-11-02): on coincidence_rule_violation_route, if a 2+ example set has no digit/mult-5/parity rule, a target outside [min,max] counts as breaking the concept (same c_viol). MAP -4008.7 vs incumbent -4027.9. Smoke fit -4017.0.
Residual finding: in incoherent sets, targets below the examples' range were over-predicted by 0.15 and above it by 0.07; inside the range was fine. Variants: separate c_span -4007.7; ramp min(gap/10,1) -4004.0; span applied to all multi sets ~0; lone-example parity/mult-5 class rule only +2.
Incoherent polarization improved: pred 0.157 vs incumbent's 0.112 (observed 0.216). Single-example polarization still unfixed.
**Why:** this targets the incoherent_multi_cell_polarization critique stat; follows [[coincidence-rule-violation-route]] and [[exp3-iter4-critique-stats]].
**How to apply:** a graded outside-span ramp gives about 4 more nats if a refinement is wanted.
