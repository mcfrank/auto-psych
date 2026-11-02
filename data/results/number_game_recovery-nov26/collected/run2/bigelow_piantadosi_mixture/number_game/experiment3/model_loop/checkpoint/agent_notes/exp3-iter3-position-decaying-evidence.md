---
name: exp3-iter3-position-decaying-evidence
description: Exp3 iter-3 cand_0 — example evidence scaled exp(rho*(k-1)) has rho≈+0.015 (null); with an equal range prior it ties the incumbent at MAP; exact sparse (set, coverage-pattern, width) class tabulation gives 14 ms/grad
metadata:
  type: project
---
Exp3 iter 3, candidate_0 (2026-10-31), exploratory order slot. Submitted `position_decaying_example_evidence`: each example's log-evidence is scaled by exp(rho*(k-1)) (total not conserved), the range prior is equal (no Weber, heading or upward terms), sampling is with replacement, plus digit sums and a lapse.
- MAP on the incumbent base (exact grid): rho=0 gives -3189.44, free rho gives -3188.65 (rho +0.02). Weber+heading+upward together are worth only about 1 nat.
- The submitted model's MAP is -3190.47 against the incumbent's -3190.30, with 3 fewer parameters. Its pool RMSE is at least 0.0059 vs the incumbent, heading_weber, digit_sum_open_ended and example_blind_lapse. The smoke fit gave -3192.4.
- So rho does not capture the prefix-extension critique: the decay comes out null.
Speed trick: under an equal range prior, an interval's evidence depends only on its (contiguous sorted-example coverage pattern, width). Flatten the nonzero (set, pattern, width) classes (~67k) and the (cell, class) entries (~268k) and use inc_subtensor segment sums. That is 14 ms/grad, exact; the dense 100x100 grid took 310 ms.
PITFALL: compile_fn on raw module-level tensors draws the RVs at random; use model.replace_rvs_by_values first.
See [[exp3-iter3-critique]], [[number-game-ceiling]].
