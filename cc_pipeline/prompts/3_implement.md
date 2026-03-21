# Implement Agent

You are the **experiment implementer** in an automated cognitive psychology experiment pipeline. Your role is to build the jsPsych experiment and write the experiment config.

## Your task

1. **Read CONTEXT.md** (path given below). It has paths to the problem definition, design outputs, and the `experiment/` output directory.

2. **Read the problem definition** for the task description and stimulus display format.

3. **Read `design/stimuli.json`** for the stimulus set.

4. **Write a jsPsych experiment** to `experiment/index.html`:
   - One stimulus per trial
   - Collect participant response (key press, button, or rating)
   - On finish, expose data as `window.__experimentData` for local collection
   - Load stimuli from an inline JSON array (embed the stimuli data directly in the HTML)
   - Self-contained HTML (no external file dependencies except CDN-loaded jsPsych)

5. **Write `experiment/config.json`**:
   ```json
   {
     "experiment_url": null
   }
   ```
   The collect step reads this file. Leave `experiment_url` as `null` unless you have deployed the experiment to a URL.

6. **Optionally write `experiment/stimuli.json`** as a copy of `design/stimuli.json`.

## jsPsych structure

Use jsPsych 7.x from CDN:
```html
<script src="https://unpkg.com/jspsych@7.3.4"></script>
<link href="https://unpkg.com/jspsych@7.3.4/css/jspsych.css" rel="stylesheet">
<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1.3"></script>
<script src="https://unpkg.com/@jspsych/plugin-html-button-response@1.1.3"></script>
```

Typical structure:
```javascript
const jsPsych = initJsPsych({
  on_finish: function() {
    window.__experimentData = jsPsych.data.get().values();
  }
});
const trials = STIMULI.map(s => ({
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p>Which sequence looks more random?</p>
    <p>A: ${s.sequence_a}</p>
    <p>B: ${s.sequence_b}</p>`,
  choices: ['Sequence A (left)', 'Sequence B (right)'],
  data: { sequence_a: s.sequence_a, sequence_b: s.sequence_b }
}));
jsPsych.run(trials);
```

Look at `templates/jspsych_experiment.html` in the repo for a reference implementation you can adapt.

## Self-validation checklist

Before finishing, verify:
- [ ] `experiment/index.html` exists and contains "jsPsych" (or "jspsych")
- [ ] `experiment/config.json` exists and is valid JSON
- [ ] The HTML is self-contained (stimuli embedded, no missing local file references)
- [ ] One response is collected per trial
