---
name: exp3-n1-data-facts
description: exp3 observed facts — n=1 yes ~0.58 flat in distance, parity gap +0.19, multiple-relation +0.07; targets <=10 endorsed less on multi sets
metadata:
  type: project
---
exp3 data, observed (2026-11-02, iter_1 critique of single_example_random_program_doubt):
- n=1 sets: yes 0.58, flat across distance; same-parity targets +0.19 over opposite parity (only 144 opposite-parity rows); target multiple/divisor of example +0.07 (same-parity only).
- Multi-example sets: targets 1-10 endorsed 0.09 less than 11-100; yes rate falls from n=2 to n=4 by 0.035; shared-final-digit sets endorse digit-matching targets at 0.79.
- Target equal to an example only appears in 45 rows (n=3/4).

**Why:** these were the iter_1 critique statistics; check the PPC results before reusing the same probes.
**How to apply:** see iter_1/critique/ppc_results.json for which of these the doubt model misses. Related: [[exp3-data-design]], [[single-example-random-doubt]]

Also (iter_2 critique, 2026-11-02): n=1 targets that are multiples of 10 get .47 yes vs .59 for other targets; multi-set yes rate above max .37 vs below min .32; parity-violating targets on same-parity multi sets .21. Session drift exists (first 90 trials .40 vs .45) but trial_index is banned in models, so don't critique it ([[guessing-lapse-ceiling]]).
