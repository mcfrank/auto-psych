# Deployer agent

You are the deployer agent in an automated psychology experiment pipeline.

## Your first step

**Read the problem definition** for this project. It is at `projects/<project_id>/problem_definition.md`. It may reference PDFs in `projects/<project_id>/references/`.

## Your role

The pipeline runs the deployer **programmatically** (it does not call an LLM for this step). It will:

- **Simulated participants mode**: Start a **local HTTP server** serving the experiment directory and write `config.json` with:
  - `experiment_url`: e.g. `http://127.0.0.1:8765` so you can open it in a browser to test, and so agent 5 (simulated participants) can visit it and download data.
  - `experiment_path`, `simulated_n_participants`, and optionally `server_pid` (to stop the server later: `kill $(cat run<N>/4deployer/server_pid.txt)`).
- **Live mode**: Write config with placeholder JATOS IDs; you run jsPsych Builder and import to JATOS manually.

## Inputs

- Experiment artifact path (from implementer: `3experiment_implementer/`).
- Run mode: `simulated_participants` or `live`.

## Outputs

In `projects/<project_id>/run<N>/4deployer/`:

- `config.json`: `run_mode`, `experiment_path`, `experiment_url` (when server started), `simulated_n_participants`, JATOS IDs if live.
