# Subjective randomness: problem definition

## Task

On each trial, the participant sees **two sequences of coin flips** (H and T) and chooses which sequence looks **more random**. This is a classic paradigm for studying representativeness and alternation biases (e.g. Griffiths, "The Rational Basis of Representativeness").

## Experiment constraints

- **Total trials per experiment: 30.** Target duration is about 5 minutes at ~5 seconds per trial (consistent with Prolific).
- **Allowed sequence lengths: 4, 6, 8.** Pairs may mix lengths (e.g. a 4-symbol sequence vs a 6-symbol sequence).

## Stimulus design schema

- **Stimulus**: A pair of two sequences of H and T. Each sequence is a string (e.g. `"HHT"`, `"HTHTHT"`, `"HHTHTTHT"`). The two sequences in a pair **may have different lengths** (e.g. length 4 vs length 6).
- **Response**: Which sequence was chosen (e.g. "left" / "right", or sequence A / sequence B).
- **Stimulus space**:
  - Allowed sequence lengths: 4, 6, 8 (see Experiment constraints).
  - Candidate stimuli: pairs of sequences (same-length or mixed-length) chosen by optimal design. The designer must select **exactly 30** stimuli per experiment.

## Suggested theories to consider

When adding theories, the theorist should take these suggestions into account (include or favor them when appropriate):

- Include **at least one theory based on the rational basis of representativeness**: a likelihood comparison between well-specified generative models of the data (e.g. Griffiths-style: compare sequences under different generative models and choose the one with higher likelihood under the preferred model).

## Theoretical models (for the theorist agent)

The following model types are in the curated library; the theorist selects and configures which to use. When the problem definition includes a "Suggested theories to consider" section, favor or include those suggestions when adding models.

1. **Bayesian (fair coin)**: Probability of sequence under fair coin; preference for the sequence with higher likelihood.
2. **Representativeness heuristic**: Preference for sequences that "look" more random (e.g. closer to 50/50 balance, more alternation).
3. **Alternation heuristic**: Preference for sequences with more H–T and T–H alternations.
4. **Subjective generator**: Infer an unknown bias; preference based on representativeness under a subjective generator model (Griffiths-style).

All models are implemented as functions that, given a stimulus (pair of sequences) and response options, return a probability distribution over responses. The experiment designer uses these to compute expected information gain and select stimuli.

## Optional references

- PDFs or papers in `references/` may be cited for scientific background (e.g. Griffiths on representativeness). Agents may read them if needed.
