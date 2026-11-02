---
name: contains-digit-cost-finding
description: Fitted extra cost on contains-digit rules gained ~17 nats MAP over multiplicative_not_exponential_concepts; tens-digit rules in place of contains hurt by 42
metadata:
  type: project
---
Round 4 (2026-10-30), cell_1/experiment1: MAP screens on multiplicative_not_exponential_concepts (base loglik -826.3).
- A fitted extra log-prior cost on the 10 contains-digit rules gave -809.4 (cost about 1.5). This addresses the critique stat shared_contained_digit_other_ending_yes. Submitted as `hidden_digit_costly_concepts`.
- Replacing the contains-digit rules with "tens digit is d" rules gave -868.5, which is much worse.
On the older erlang base the same contains cost gave only +2.6 (see [[incumbent-null-changes-round3]]), so the gain depends on the base model.

**Why:** saves re-screening digit-rule variants.
**How to apply:** keep the contains rules and price them. Screen with the fit2.py script in iter_3/candidate_0/scratch. See [[exponent-concept-cost-finding]].
