# Critique of `privileged_named_number_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **digit_rule_size_principle** — Among multi-example sets whose examples all share one final digit, yes-rate on targets NOT matching that digit for sets of >=3 examples minus for 2-example sets; observed < null means people reject rule-violating targets more strongly as consistent examples accumulate (a size principle) than the model, whose mutual-similarity shortfall is identical for 2 or 4 uniform examples; observed > null the reverse. (observed -0.134 vs null mean 0.0374, z=-3.97, p=0.002, q=0.016) [survives FDR]
- **repdigit_concept_boost** — In sets containing at least one repdigit example (11,22,...,99), yes-rate on repdigit targets minus yes-rate on non-repdigit targets that share a final digit with some example (examples themselves excluded); observed > null means the model under-produces a "same-digit numbers" concept, observed < null that it over-credits it. (observed 0.0666 vs null mean -0.063, z=2.14, p=0.034, q=0.136)
