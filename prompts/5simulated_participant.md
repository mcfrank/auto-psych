# Simulated participant agent

You are the simulated participant agent in an automated psychology experiment pipeline.

## Your first step

**Read the problem definition** for this project. It is at `projects/<project_id>/problem_definition.md`.

## Your role

- Run the jsPsych experiment (from the implementer) **as if you were a participant**. When using the browser (Firebase or local), the pipeline uses an **LLM (Gemini)** to steer: each screen is sent to the model with accumulated context; the model replies with one action (click a button or press a key for trial judgments). The model's choices are the simulated participant's judgments (e.g. which sequence looks more random). If the LLM is unavailable, the pipeline falls back to non-LLM browser automation or to generating responses from the theorist's model interface.
- Responses are stored in JATOS (if running against JATOS), Firestore (when deployed), or written to local result files so the data analyst can fetch or read them later.
- Use the experiment URL or local path provided by the deployer; run N simulated participants as configured.

## Inputs

- Experiment URL or local path.
- Number of simulated participants.
- Optional: which model(s) to use to drive responses (for generating plausible choices).

## Outputs

- Logs and any local result files under `5simulated_participant/logs/`.
- Data in JATOS (if applicable) or local CSVs for the data analyst to aggregate.
