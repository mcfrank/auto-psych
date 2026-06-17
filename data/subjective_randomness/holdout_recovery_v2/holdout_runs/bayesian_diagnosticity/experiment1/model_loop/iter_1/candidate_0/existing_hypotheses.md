# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.296, ELPD-LOO -414.22

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.004, ELPD-LOO -418.15

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## iter0_candidate0  — posterior 0.603, ELPD-LOO -413.65

People judge a sequence as random-looking based entirely on how close its alternation rate is to an internal prototype ideal — the frequency at which a truly random sequence ought to switch between heads and tails. A sequence that alternates at the right rate looks most random; sequences that are either too streaky or too repetitively alternating look less random. The overall balance of heads versus tails in the sequence provides no additional information once alternation rate is accounted for.

## iter0_candidate1  — posterior 0.044, ELPD-LOO -416.47

People judge a sequence as more random when its longest streak of identical outcomes is shorter. A long run of repeated heads or tails is the single most cognitively salient violation of randomness; sequences where the maximum run is brief look random, and sequences containing even one long streak look non-random. No other feature of the sequence — overall head balance, alternation rate, or periodic structure — matters once maximum run length is accounted for.

## iter0_candidate2  — posterior 0.054, ELPD-LOO -416.32

People judge a sequence as more random when its proportion of heads and tails is closer to 50/50. Outcome balance is the single salient cue — a fair coin should produce roughly equal numbers of each outcome, so a sequence that deviates from that balance looks non-random. No other feature of the sequence, such as its switching rate or streak length, matters beyond this balance check.
