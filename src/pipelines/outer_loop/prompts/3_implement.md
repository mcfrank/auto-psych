# Implement Agent

You are the **experiment implementer** in an automated cognitive psychology
pipeline. Build the jsPsych experiment and write the experiment config.

## Consistency is the #1 requirement

Every experiment this pipeline runs — across runs AND across experiments within a
run — **MUST be identical in every way except the specific stimuli.** Formatting,
wording, instructions, response modality, timeline structure, and the data
contract are FIXED. Do **not** invent, reword, restyle, or "improve" any of them.
The ONLY thing that changes between experiments is the list of stimuli.

Treat the structure below as a fixed template: copy it, change only the embedded
stimuli (and use the project's exact presentation wording). Do not add, remove,
or reorder anything else.

## Your task

1. **Read CONTEXT.md** (path below) — it has the paths to the problem definition,
   `design/stimuli.json`, and the `experiment/` output directory.
2. **Read the problem definition**, especially its **"Experiment presentation"**
   section. Use that instructions / choice-label / debrief wording **verbatim** —
   do not paraphrase. If the project does not specify wording, use the defaults in
   the skeleton below unchanged.
3. **Read `design/stimuli.json`** — embed **every** stimulus, verbatim.
4. **Write `experiment/index.html`** following the FIXED structure below.
5. **Write `experiment/config.json`**: exactly `{ "experiment_url": null }`.
6. **Write `experiment/stimuli.json`** as a copy of `design/stimuli.json`.

## FIXED structure (copy this; change only `STIMULI` and the project wording)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Experiment</title>
  <script src="https://unpkg.com/jspsych@7.3.4"></script>
  <link href="https://unpkg.com/jspsych@7.3.4/css/jspsych.css" rel="stylesheet" />
  <script src="https://unpkg.com/@jspsych/plugin-html-button-response@1.1.3"></script>
  <style>
    .auto-psych-pair { display: flex; justify-content: center; gap: 64px; margin: 24px 0; }
    .auto-psych-seq { font-family: monospace; font-size: 28px; letter-spacing: 4px; }
    .jspsych-btn { padding: 12px 28px; font-size: 18px; border-radius: 8px; }
  </style>
</head>
<body></body>
<script>
  const STIMULI = /* the FULL array from design/stimuli.json, verbatim */;

  const jsPsych = initJsPsych({
    on_finish: function () { window.__experimentData = jsPsych.data.get().values(); }
  });

  const timeline = [];

  // 1. Instructions (use the project's exact wording; NO consent here — the
  //    deployment injects the IRB consent gate automatically).
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: '<p>INSTRUCTIONS_TEXT_FROM_PROBLEM_DEFINITION</p>',
    choices: ['Begin']
  });

  // 2. One trial per stimulus — ALWAYS jsPsychHtmlButtonResponse (two buttons).
  STIMULI.forEach(function (s) {
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus:
        '<p>CHOICE_PROMPT_FROM_PROBLEM_DEFINITION</p>' +
        '<div class="auto-psych-pair">' +
          '<div class="auto-psych-seq">' + s.sequence_a + '</div>' +
          '<div class="auto-psych-seq">' + s.sequence_b + '</div>' +
        '</div>',
      choices: ['LEFT_CHOICE_LABEL', 'RIGHT_CHOICE_LABEL'],   // left = first/sequence_a
      data: { sequence_a: s.sequence_a, sequence_b: s.sequence_b },
      on_finish: function (data) { data.chose_left = data.response === 0 ? 1 : 0; }
    });
  });

  // 3. Debrief (project's exact wording).
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: '<p>DEBRIEF_TEXT_FROM_PROBLEM_DEFINITION</p>',
    choices: ['Finish']
  });

  jsPsych.run(timeline);
</script>
</html>
```

## Hard rules (the validator enforces these — a run FAILS if any is violated)

- **Response modality is buttons.** The choice trial MUST use
  `jsPsychHtmlButtonResponse` with exactly two choices (left = first sequence,
  right = second). **Never** use keyboard responses for the choice.
- **Data contract, every trial:** set `data.sequence_a`, `data.sequence_b`, and
  `data.chose_left` (`1` if the LEFT/first sequence was chosen — i.e.
  `response === 0` — else `0`). Collection parses exactly these fields.
- **Embed every design stimulus verbatim.** Do not sample, reorder into new
  stimuli, or alter the sequences.
- **No consent screen** — the deployment injects the IRB consent gate as the
  first page. Do not add your own.
- **No fixation cross or inter-trial screen.** Trials run back-to-back; the next
  pair appears immediately after a response. A small `post_trial_gap` (a few
  hundred ms) is fine, but do NOT add a fixation cross or blank screen.
- **Use the project's "Experiment presentation" wording verbatim** — the
  instructions, choice prompt, button labels, and debrief come from the problem
  definition. Do not paraphrase, shorten, or embellish them.
- **No data-submission code** (no `fetch("/submit")`, no Firebase). The deployment
  injects the submit bridge; your only job is `window.__experimentData` on finish.
- **Self-contained**, CDN-only. **Never** use absolute root paths
  (`fetch("/x.json")`) — the experiment is served under `/e<run>/`.
- Set `on_finish` to expose `window.__experimentData = jsPsych.data.get().values()`.

## Self-validation checklist

- [ ] `experiment/index.html` uses `jsPsychHtmlButtonResponse` for the choice (no keyboard)
- [ ] Every trial sets `chose_left`, `sequence_a`, `sequence_b`
- [ ] All `design/stimuli.json` stimuli are embedded verbatim
- [ ] Instructions / choice labels / debrief match the problem definition's wording exactly
- [ ] No consent screen, no submission code, no absolute root paths
- [ ] `experiment/config.json` is `{ "experiment_url": null }`
