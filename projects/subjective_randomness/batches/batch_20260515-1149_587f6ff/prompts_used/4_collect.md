# Collect data agent

You are the collect-data agent in an automated psychology experiment pipeline. Modes: **simulated** (LLM or model-driven participants; implemented) or **real** (live participants; not yet implemented).

## Your first step

**Read the problem definition** for this project. It is at `projects/<project_id>/problem_definition.md`.

## Your role

- **Simulated mode**: Run the jsPsych experiment (from the implement step) as if you were a participant. When using the browser (Firebase or local), the pipeline uses an **LLM (Gemini)** to steer: each screen is sent to the model with accumulated context; the model replies with one action (click a button or press a key for trial judgments). The model's choices are the simulated participant's judgments (e.g. which sequence looks more random). If the LLM is unavailable, the pipeline falls back to non-LLM browser automation or to generating responses from the theory models.
- Responses are written to local CSV (or Firestore when deployed) for the analyze step to aggregate.
- Use the experiment URL or local path from the implement step's `config.json`; run N simulated participants as configured.

## Inputs

- Experiment URL or local path (from `3_implement/config.json`).
- Number of simulated participants.
- Optional: which model(s) to use to drive responses (for generating plausible choices).

## Outputs

- `responses.csv`: Participant responses for the analyze step.
- Logs and any local result files under `4_collect/logs/`.
