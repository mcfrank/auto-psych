"""A jsPsych reference-game experiment for the RSA pipeline.

* `design` turns abstract contexts (object x feature matrices, a heard
  feature or a mumble) into concrete per-participant trial lists: item
  domain, feature words, screen order, base images, catch trials.
* `build` renders `template.html` with those lists into a static site (or a
  single-file preview with the images inlined).
* `convert` turns the page's jsPsych data back into rows of the canonical
  trial schema (`src/pipelines/outer_loop/projects/rsa_reference/data/README.md`),
  which `src.rsa.dataset.context_from_row` loads.

The stimuli are the pragmods images (langcog/pragmods_expts), copied to
`src/pipelines/outer_loop/projects/rsa_reference/experiment/images/`.
"""
