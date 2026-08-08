# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## minkowski_accumulated_typicality  — posterior 0.000, ELPD-LOO -809.40

People evaluate the randomness of a sequence by accumulating a subjective
sense of typicality over its length, computing each event's typicality as
a penalty based on its distance from a mental prototype, with the distance
raised to a freely inferred Minkowski-like exponent so extreme feature
deviations are disproportionately punished.

## evidence_accumulation_messy_prototype  — posterior 0.000, ELPD-LOO -835.54

Random-looking sequences are judged by an evidence-accumulation process
where each item provides a baseline weight of evidence for randomness,
discounted by the sequence's quadratic (and asymmetric, for alternation)
deviation from a messy prototype — an ideal positive imbalance and ideal
alternation rate — so near-ideal longer sequences are preferred while
clearly deviant long sequences are penalized more heavily.

## evidence_accumulation_per_run  — posterior 1.000, ELPD-LOO -782.39

Random-looking sequences are judged by an evidence-accumulation process
where each distinct run (streak of identical outcomes) — rather than each
item — provides a baseline weight of evidence for randomness, discounted
by the sequence's quadratic deviation from a messy prototype, so periodic
patterns with artificially few runs are penalized without a separate
periodicity cue.

## artificial_balance_diagnosticity  — posterior 0.000, ELPD-LOO -825.29

People judge randomness by Bayesian diagnosticity — the log-likelihood
ratio of a fair coin versus a subjective "regular" generative process —
where the regular process includes an "artificial balance" generator that
targets an exactly equal count of heads and tails, explaining why people
penalize suspiciously symmetric sequences as contrived rather than random.
