# Auto-psych: LangGraph agentic experiment loop

A 6-agent pipeline for psychology experiments: theory → design → implement (includes deploy) → collect data → analyze → interpret. State is file-based under `projects/<phenomenon>/run<N>/<agent>/`.

## Agents

| # | Agent | Role | Key outputs |
|---|-------|------|-------------|
| 1 | **Theory** | Adds **one theory per LLM call**. Run 1: 2–3 calls to add 2–3 models; run 2+: copies previous run’s theories and adds at least one new (1+ calls). Each call outputs one YAML model + one Python file, then ---DONE--- or ---ADD_ANOTHER---. | `1_theory/models_manifest.yaml`, `1_theory/*.py`, `model_registry.yaml` (run dir) |
| 2 | **Design** | Generates a design script that scores stimuli by expected information gain (EIG) using the registry’s theory probabilities; selects top stimuli. | `2_design/stimuli.json`, `design_rationale.md` |
| 3 | **Implement** | Fills the jsPsych template; then runs deploy step (local server or Firebase), writes config. | `3_implement/index.html`, `stimuli.json`, `config.json` |
| 4 | **Collect data** | Simulated mode: opens the experiment in a browser (or Firebase), collects responses and writes CSV. Real mode not yet implemented. | `4_collect/responses.csv` |
| 5 | **Analyze** | Aggregates responses by stimulus, computes summary statistics (Python only). | `5_analyze/aggregate.csv`, `summary_stats.json` |
| 6 | **Interpret** | Compares data (merged over runs 1..n) to model predictions; writes a report and updates theory probabilities in the run's registry. | `6_interpret/report.md`, `theory_probabilities.yaml`, `model_registry.yaml` (updated) |

After each agent runs, a **validation** step runs automatically. If validation fails, the agent is re-invoked with feedback (up to 3 retries), then the pipeline continues to the next agent.

### Observability and debugging

Each agent writes to its run directory:

- **`observability.log`** — Timestamped log of what the agent did: start/end, validation feedback (on retries), LLM invocation, file writes, and any errors. When validation fails, the validator also appends the failure reason here.
- **`transcripts/`** — For agents that call the LLM (theory, design, interpret), one Markdown file per attempt: `transcripts/attempt_001.md`, etc. Each file contains the system prompt, user message, and full LLM response (and any validation feedback for that attempt), so you can see exactly what was sent and why it might have failed or timed out.

The theory agent is called iteratively (one model per call) to reduce validation failures; each call has a 5-minute timeout.

## Requirements

- **Python 3.11+** (recommended)
- **Node.js 14+** (optional; for jsPsych Builder when building/packaging experiments)
- **Firebase** (optional): for collect step with no local server — Hosting + Cloud Functions + Firestore. Otherwise the pipeline uses a local HTTP server.

## Installation

### 1. Clone and enter the project

```bash
cd auto-psych
```

### 2. Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Secrets and config files

Three different things are involved; only one holds API keys.

| File | Purpose | Secret? | Do you need it? |
|------|--------|--------|------------------|
| **`.secrets`** | Holds **API keys** (and optional overrides) the pipeline reads at runtime. | Yes — gitignored. | Yes. Create it and add `GOOGLE_API_KEY` for the LLM agents. |
| **`.firebaserc`** | Tells the **Firebase CLI** which Firebase project to use (`default` project ID). | No — project ID is not secret. | Only if you use Firebase deploy; set `"default": "your-firebase-project-id"`. |
| **`firebase.json`** | **Firebase project config**: hosting folder, rewrites to Cloud Functions. | No. | Yes if you use Firebase; the repo includes a default. Set `hosting.site` to your project ID when deploying manually. |

**What to put in `.secrets`**

- **`GOOGLE_API_KEY`** (required for theory, design, implement, collect steering, interpret): Get an API key for Gemini from [Google AI Studio](https://aistudio.google.com/apikey). The pipeline reads this from the `.secrets` file or from the `GOOGLE_API_KEY` environment variable.
- **`PROLIFIC_API_TOKEN`** (required for `--mode test_prolific` or live Prolific): Create a token in [Prolific Researcher Settings](https://app.prolific.com/researcher/). Used to create test participants and studies via the [Prolific API](https://docs.prolific.com/api-reference/).

Example `.secrets` (do not commit):

```
GOOGLE_API_KEY=your_gemini_api_key_here
PROLIFIC_API_TOKEN=your_prolific_token_here
```

You can use a `.secrets` directory with one file per key (e.g. `.secrets/GOOGLE_API_KEY`) if you prefer.

**Firebase:** The deploy step does **not** read any Firebase API key from `.secrets`. Firebase deploy uses the CLI and your login (`firebase login`). The project is chosen from `.firebaserc` (or the `FIREBASE_PROJECT` environment variable). So you need `.firebaserc` (or `FIREBASE_PROJECT`) and `firebase.json` for Firebase; no Firebase keys in `.secrets`.

### 4. (Optional) Firebase for collect (simulated) with no local server

To run simulated participants against a **deployed experiment** (no local server):

1. **Firebase project**: Create a project at [Firebase Console](https://console.firebase.google.com/), enable Hosting and Firestore.
2. **CLI**: `npm install -g firebase-tools` then `firebase login`. In the repo root, set your project in `.firebaserc`: `"default": "your-project-id"` (or set env `FIREBASE_PROJECT`). If you deploy with `firebase deploy` manually, set the same project ID in `firebase.json` under `hosting.site` (the implement step sets this for you).
3. **Deploy**: Run the pipeline with `--mode simulated_participants`. The implement step copies the experiment to `public/`, deploys with `firebase deploy --only hosting,functions`, and writes `experiment_url` and `results_api_url` in `3_implement/config.json`. The collect step runs simulated (browser or model sampling) or live (Prolific + Firebase when implemented); in simulated mode it opens the experiment URL N times (each run POSTs to `/submit`), then fetches GET `/results` and writes `responses.csv`.

If Firebase is not configured (no real project in `.firebaserc`), the deploy step uses the **local server** at `http://127.0.0.1:8765` and the collect step gathers data from `window.__experimentData` in the browser.

### 5. (Optional) Test Prolific end-to-end

To test the Prolific flow with a single test participant (no real recruitment):

1. **Prolific API token**: Add `PROLIFIC_API_TOKEN` to `.secrets` (see above). The [test participant API](https://docs.prolific.com/api-reference/users/create-test-participant-for-researcher) must be enabled for your Prolific workspace.
2. **Project config**: Create `projects/<project_id>/prolific_config.yaml` (or copy from `prolific_config.yaml.example`). Set **`test_participant_email`** to an email that is **not** already registered on Prolific; this account will be created as a test participant and used as the only place in the study.
3. **Run**: `python3 run_pipeline.py --project <project_id> --run 1 --mode test_prolific`. The implement step deploys to Firebase, creates the test participant, creates a 1-place study with that participant on a custom allowlist, and publishes it. The collect step polls Prolific until the study is complete, then fetches results from Firebase. After the participant finishes the experiment, the page redirects to Prolific’s completion URL.

## Running the pipeline

```bash
source venv/bin/activate
python3 run_pipeline.py --project subjective_randomness --run 1 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 3 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 4-6 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 2 --n-participants 10 --max-retries 5
```

- `--project`: Project id (e.g. `subjective_randomness`); must have a `problem_definition.md` under `projects/<project>/`.
- `--run`: Single run number (creates `projects/<project>/run<N>/`). Use this or `--runs`.
- `--runs`: Runs to execute: **N** (runs 1 through N) or **A-B** (runs A through B inclusive, e.g. `4-6`). Each run gets its own directory; the theory agent in run n reads run n−1’s interpreter report and registry; the interpreter in run n sees merged data from runs 1..n.
- `--mode`: `simulated_participants`, `live`, or `test_prolific`. Use `test_prolific` to run the full Prolific flow with one test participant (Firebase deploy + Prolific test participant + study + collect).
- `--n-participants`: Number of simulated participants per run (default: 5). Default is in `src/config.py` as `DEFAULT_SIMULATED_N_PARTICIPANTS`.
- `--max-retries`: Max validation retries per agent before moving on (default: 3). Default in `src/config.py` as `DEFAULT_MAX_VALIDATION_RETRIES`.

Validation runs **inside** the pipeline after each agent: if an agent’s output fails validation, the agent is re-run with the failure message (up to `--max-retries`), then execution continues.

Prompts are resolved from `prompts/` with optional overrides in `projects/<project>/prompts/`. The resolved prompts used for each run are copied to `projects/<project>/run<N>/prompts_used/` for reproducibility.

## Running a single agent (debugging)

To run one agent in isolation (no validation loop, no next agent), use **`run_pipeline.py --agent`** or the convenience script **`run_agent.py`** (same options):

```bash
python3 run_pipeline.py --project subjective_randomness --run 1 --agent 1_theory
python3 run_agent.py --project subjective_randomness --run 2 --agent 2_design --state-from-run 1
python3 run_agent.py --project subjective_randomness --run 1 --agent 1_theory --use-fixtures
```

- `--agent`: Agent to run (e.g. `1_theory`, `2_design`, `3_implement`, `4_collect`, `5_analyze`, `6_interpret`).
- `--state-from-run R`: Load artifact paths from run `R` so you can re-run or run a downstream agent using outputs from a previous run.
- `--use-fixtures`: Use minimal state with `tests/fixtures/` for missing inputs (no prior full run required).
- `--n-participants`, `--max-retries`: Same as pipeline (used when running implement or for consistency).

Use `python3` in the examples; on many systems `python` points to Python 2 or is missing. After `source venv/bin/activate`, `python` usually works too (venv’s `python` is Python 3).

`run_agent.py` is a thin wrapper that calls `run_pipeline.py`; all parameters are defined in `run_pipeline.py`.

## Validating outputs (standalone critic)

Validation is **built into the pipeline**: after each agent, the same checks run automatically and failed agents are re-invoked with feedback (see Agents above). You can also run the validators **standalone** (e.g. to audit a past run without re-running agents):

```bash
python3 run_critic.py --project subjective_randomness --run 1
python3 run_critic.py --project subjective_randomness --run 1 --agent 1_theory
```

- Exits 0 if all validations pass, 1 if any fail.
- Use `--no-write` to only print results and not write `validation.json` under each agent dir.
- The same validator logic is used by the pipeline’s built-in validation loop.

## Tests

Install dev dependencies and run pytest:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests include: model library (each model returns a probability distribution), theory/design output validators, state loader, and a single-agent run (theory with fixtures) followed by validation.

## Project layout

```
prompts/                 # Canonical agent prompts (one file per agent)
projects/
  <project_id>/
    problem_definition.md   # Human-authored: task + stimulus schema
    prolific_config.yaml    # Optional; for test_prolific/live (completion_code, test_participant_email, etc.)
    references/             # Optional PDFs referenced by problem_definition.md
    prompts/                # Optional project-specific prompt overrides
    run<N>/
      prompts_used/        # Frozen prompts used in this run
      1_theory/
      2_design/
      3_implement/
      4_collect/
      5_analyze/
      6_interpret/
.secrets                  # Gitignored; see above
```

## Observability

- **Prompts used per run**: For each run, the resolved prompts (canonical + any project overrides) are copied to `projects/<project_id>/run<N>/prompts_used/` (one `.md` per agent and a `manifest.json`). This gives a reproducible record of which prompt text each agent used.
- **Artifacts**: Each agent writes its outputs under `projects/<project_id>/run<N>/<agent>/` (e.g. `1_theory/models_manifest.yaml`, `5_analyze/aggregate.csv`). Inspect these to debug or reproduce a run.
- **Design agent**: Writes `2_design/designer_observability.log` with timestamped steps (LLM call, fenced-block extraction, script run, fallback reason). If the design step falls back, see that log and `design_script_log.txt` (script stdout/stderr or exception) for the cause.

## Code style and docs

- Python: Black/Ruff recommended; docstrings for public APIs.
- README and in-code comments document setup and usage.
