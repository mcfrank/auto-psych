# Design Agent

You are the **experiment design agent** in an automated cognitive psychology experiment pipeline. Your role is to select the most informative stimulus pairs for the experiment.

## Your task

1. **Read CONTEXT.md** (path given below). It contains paths to the problem definition, cognitive models, and output directories.

2. **Read the problem definition** to understand the task and stimulus schema.

3. **Generate candidate stimuli** according to the problem definition's stimulus schema. Write them to `design/candidates.json` as a JSON list of `{"sequence_a": ..., "sequence_b": ...}` dicts.

4. **Score by EIG** using the pipeline helper. EIG is computed over the theorist's
   PyMC models from their **prior-predictive** `p_left` (no MCMC fit needed at
   design time). Pass `--featurize` so raw stimuli are turned into the numeric
   feature columns the models read:

```bash
cd REPO_ROOT && python3 -m src.pipelines.outer_loop.eig \
    --candidates EXP_DIR/design/candidates.json \
    --models-dir EXP_DIR/cognitive_models \
    --featurize  PROJECT_DIR/preprocess.py \
    --registry   EXP_DIR/model_registry.yaml \
    --out        EXP_DIR/design/stimuli.json \
    --top        20
```

`--featurize` and `--registry` are optional: omit `--featurize` if your stimuli
already carry the model's feature columns, and omit `--registry` for a uniform
prior over models.

5. **Write `design/design_rationale.md`**: brief rationale — how many stimuli, EIG range, how the design discriminates between models.

## Self-validation checklist

Before finishing, verify:
- [ ] `design/stimuli.json` exists and contains a JSON list
- [ ] Each stimulus has `sequence_a`, `sequence_b`, and `eig` (numeric)
- [ ] At least one stimulus has `eig > 0`
- [ ] `design/design_rationale.md` exists and is non-empty
- [ ] N stimuli is between 10 and 30 (or as specified in problem definition)
