# Experiment templates

All experiments in this framework present the participant with: consent → instructions → trials (one per stimulus, random order) → debrief. **Consent is not part of the timeline** — the deployment step injects it as a gate (from `consent.txt`, verbatim) in front of the experiment. So the template/timeline itself starts at **instructions**. The remaining **content** is filled in by the pipeline (and optionally customized by the implementer agent).

- **`jspsych_experiment.html`** — Generic jsPsych 8 template. Placeholders:
  - `{{INSTRUCTIONS_HTML}}` — Full instructions screen HTML (built from problem definition + experiment-specific text). This is the **first** screen; do not put consent text here.
  - `{{STIMULI_JSON}}` — Array of trial objects. Each must have `stimulus_display` (HTML string for that trial) and any data keys (e.g. `sequence_a`, `sequence_b`). Precomputing `stimulus_display` in the implementer avoids runtime errors (no `.split()` on timeline variables in the browser).
  - `{{TRIAL_CHOICES_JSON}}` — Keys allowed per trial (e.g. `["f","j","ArrowLeft","ArrowRight"]`).
  - `{{DEBRIEF_HTML}}` — Debrief screen HTML.

- **`consent.txt`** — The consent form text. Edit this file to set your real, IRB-approved consent wording; at deploy time the pipeline converts it to HTML (paragraphs) and injects it verbatim as a full-screen consent gate ("I agree") in front of the experiment. This is the **only** consent participants see.

- **`jspsych_experiment_preview.html`** — Same structure as the generic template, with sample data and precomputed `stimulus_display`. Open in a browser to preview the flow without running the pipeline.

The pipeline uses **`jspsych_experiment.html`** and **`consent.txt`**; the implementer agent plugs in the experiment-specific instructions and builds `stimulus_display` for each trial (e.g. for subjective randomness: left/right sequences and response prompt).

Based on the [jsPsych Simple RT Task tutorial](https://www.jspsych.org/latest/tutorials/rt-task/).
