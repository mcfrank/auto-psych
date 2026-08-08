# Design Rationale

This design was generated using the **exhaustive** selection strategy, as the models currently share a uniform prior in this first experiment (`reserved_for_new: 0.25`, with the remaining probability mass distributed evenly across the 4 competing cognitive models). 

Exhaustively enumerating the full candidate space (all permutations of lengths 2–8, yielding 128,778 distinct pairs) ensures that the selection maximizes joint Expected Information Gain (EIG) and prevents missing critical regions of model disagreement that could result from smaller, hand-authored candidate clusters.

- **Number of stimuli**: 32
- **EIG Range**: 0.0537 – 0.4137
- **Targeting**: The objective greedily maximized joint EIG across all four models (`minkowski_accumulated_typicality`, `evidence_accumulation_messy_prototype`, `evidence_accumulation_per_run`, and `artificial_balance_diagnosticity`). Because no models are strongly weighted yet, this provides broad, maximally informative coverage across the full space of disagreements (e.g., contrasting short vs long sequence preferences, structured alternating vs imbalanced runs, and artificially balanced motifs vs messy prototypes) without manually restricting the candidate pool.
