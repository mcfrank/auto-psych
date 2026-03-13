# Experiment templates

All experiments in this framework use the same **structure**: consent → instructions → trials (one per stimulus, random order) → debrief. The **content** is filled in by the pipeline (and optionally customized by the implementer agent).

- **`jspsych_experiment.html`** — Generic jsPsych 8 template. Placeholders:
  - `{{CONSENT_HTML}}` — Consent screen HTML (from `consent.txt` by default).
  - `{{INSTRUCTIONS_HTML}}` — Full instructions screen HTML (built from problem definition + experiment-specific text).
  - `{{STIMULI_JSON}}` — Array of trial objects. Each must have `stimulus_display` (HTML string for that trial) and any data keys (e.g. `sequence_a`, `sequence_b`). Precomputing `stimulus_display` in the implementer avoids runtime errors (no `.split()` on timeline variables in the browser).
  - `{{TRIAL_CHOICES_JSON}}` — Keys allowed per trial (e.g. `["f","j","ArrowLeft","ArrowRight"]`).
  - `{{DEBRIEF_HTML}}` — Debrief screen HTML.

- **`consent.txt`** — Stub consent text. Edit this file to add your real consent form; the pipeline converts it to HTML (paragraphs) and injects it into the template.

- **`jspsych_experiment_preview.html`** — Same structure as the generic template, with sample data and precomputed `stimulus_display`. Open in a browser to preview the flow without running the pipeline.

The pipeline uses **`jspsych_experiment.html`** and **`consent.txt`**; the implementer agent plugs in the experiment-specific instructions and builds `stimulus_display` for each trial (e.g. for subjective randomness: left/right sequences and response prompt).

Based on the [jsPsych Simple RT Task tutorial](https://www.jspsych.org/latest/tutorials/rt-task/).
