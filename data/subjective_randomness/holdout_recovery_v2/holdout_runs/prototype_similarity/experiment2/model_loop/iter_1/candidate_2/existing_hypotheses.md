# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## encoding_compressibility  — posterior 0.183, ELPD-LOO -509.93

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.182, ELPD-LOO -508.64

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.304, ELPD-LOO -507.82

Best PyMC model found by the inner model-improvement loop.

## alternation_prototype  — posterior 0.000, ELPD-LOO -544.10

People judge a sequence as more random when its alternation rate is closer
(L1 distance) to an internalized prototype value that is biased above 0.5,
reflecting the well-documented human tendency to overestimate alternation in
random sequences.

## iter0_candidate0  — posterior 0.327, ELPD-LOO -507.80

People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky. While the alternating generator has a canonical switch probability (0.95), people's mental prototype for streakiness is flexible: the characteristic persistence of the streaky generator (its switch probability) is a learned cognitive parameter rather than a fixed constant at 0.15.

## iter0_candidate1  — posterior 0.005, ELPD-LOO -514.58

People judge a sequence as more random when its proportion of heads is closer to 0.5. When comparing two sequences, they choose the one whose head count deviates less from an equal split as more random — imbalance is the sole cue driving the choice.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -564.14

People judge a sequence as more random when its longest unbroken run of identical outcomes is shorter. The maximum run length is the sole cue: a long streak signals a non-random, streaky process, so people pick whichever sequence has the shorter maximum run as looking more random.

## iter1_candidate0

People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky. While the alternating and streaky generators are defined by canonical transition probabilities, people's mental model of what a biased sequence looks like is flexible: the characteristic head probability of the biased generator is a learned cognitive parameter rather than a fixed constant.

## iter1_candidate1

People judge a sequence as more random when it contains less detectable periodic structure. When comparing two sequences, they choose the one with lower periodicity as the more random-looking one — periodic regularity signals a non-random, patterned generator.
