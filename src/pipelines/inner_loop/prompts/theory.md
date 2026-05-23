# Cognitive Model Candidate Generation

You are generating one candidate cognitive model for an auto-psych experiment.

Read these files in the current working directory before deciding what to change:

1. `CONTEXT.md`
2. `CANDIDATE_BRIEF.md`
3. Any compact diagnostics files, especially `history.json`, `incumbent_fit.json`,
   `per_trial_metrics.json`, `summary_stats.json`, `examples.md`, and `round_delta.md`.

## Goal

Write a full replacement `cognitive_model.py` that improves fit to observed
choice data. The first target domain is subjective randomness: choosing which
of two binary sequences looks more random.

## Required Output

Write `cognitive_model.py` in the current working directory. It must define:

- `PARAM_NAMES`
- `PARAM_BOUNDS`
- `INITIAL_PARAMS`
- `cognitive_model(stimulus, response_options, params=None) -> dict[str, float]`

For subjective randomness, `stimulus` is a tuple like:

```python
("HHTHTTHT", "HTHTHTHT")
```

and `response_options` is usually:

```python
["left", "right"]
```

Return probabilities over response options:

```python
return {"left": p_left, "right": 1.0 - p_left}
```

## Constraints

- Keep parameters continuous floats so they can be fit downstream.
- Clip params to bounds inside the function.
- Probabilities must be finite and sum to 1.0.
- Keep the model interpretable: feature scores, softmax/logistic choice rules,
  mixtures, lapse rates, and hierarchical/PyMC-compatible parameterizations are
  all acceptable.
- Do not import project-private modules except stable auto-psych helpers.

## Self-check

Before stopping, make sure this succeeds:

```python
result = cognitive_model(("HHTHTTHT", "HTHTHTHT"), ["left", "right"], INITIAL_PARAMS)
assert set(result) == {"left", "right"}
assert abs(sum(result.values()) - 1.0) < 1e-6
```
