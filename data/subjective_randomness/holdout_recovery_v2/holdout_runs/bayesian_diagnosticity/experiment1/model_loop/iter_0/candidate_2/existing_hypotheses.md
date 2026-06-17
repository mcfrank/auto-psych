# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.988, ELPD-LOO -414.22

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.012, ELPD-LOO -418.15

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## iter0_candidate0

People judge a sequence as random-looking based entirely on how close its alternation rate is to an internal prototype ideal — the frequency at which a truly random sequence ought to switch between heads and tails. A sequence that alternates at the right rate looks most random; sequences that are either too streaky or too repetitively alternating look less random. The overall balance of heads versus tails in the sequence provides no additional information once alternation rate is accounted for.

## iter0_candidate1

People judge a sequence as more random when its longest streak of identical outcomes is shorter. A long run of repeated heads or tails is the single most cognitively salient violation of randomness; sequences where the maximum run is brief look random, and sequences containing even one long streak look non-random. No other feature of the sequence — overall head balance, alternation rate, or periodic structure — matters once maximum run length is accounted for.
