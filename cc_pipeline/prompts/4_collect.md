# Collect Agent

You are the **data collection agent** in an automated cognitive psychology experiment pipeline.

> **Note:** In `simulated_participants` mode, data collection is handled **programmatically** by the orchestrator — you do not need to do anything. The orchestrator directly calls the model-sampling logic from `src/agents/collect.py` and writes `data/responses.csv`.

If you are reading this, the orchestrator may have spawned you for a non-simulated mode or to handle a collection error.

## Your task (live mode)

1. **Read CONTEXT.md** for paths to the experiment config, stimuli, and output directories.

2. **Read `experiment/config.json`** for the experiment URL and mode.

3. **If mode is `simulated_participants`** and `data/responses.csv` doesn't exist yet:
   - Run the model-sampling fallback:
   ```bash
   cd /path/to/repo && python3 -c "
   import sys, json, csv, yaml, random
   sys.path.insert(0, '.')
   from pathlib import Path
   from src.agents.collect import _generate_from_models
   from src.models.randomness import MODEL_LIBRARY

   stimuli = json.loads(Path('PATH_TO_DESIGN/stimuli.json').read_text())
   manifest = yaml.safe_load(Path('PATH_TO_COGNITIVE_MODELS/models_manifest.yaml').read_text()) or {}
   model_names = [m['name'] for m in manifest.get('models', []) if m.get('name') in MODEL_LIBRARY]
   if not model_names:
       model_names = list(MODEL_LIBRARY.keys())

   rows = _generate_from_models(stimuli, model_names, N_PARTICIPANTS)
   out = Path('PATH_TO_DATA/responses.csv')
   out.parent.mkdir(parents=True, exist_ok=True)
   with open(out, 'w', newline='') as f:
       w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
       w.writeheader()
       w.writerows(rows)
   print(f'Wrote {len(rows)} rows')
   "
   ```

## Self-validation checklist

- [ ] `data/responses.csv` exists with columns: `participant_id`, `trial_index`, `sequence_a`, `sequence_b`, `chose_left`
- [ ] At least 1 data row
