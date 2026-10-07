# Reference-game experiment assets

Assets of the jsPsych reference-game experiment (`src/rsa/experiment/`).

## `images/` — provenance

The stimuli of the pragmods experiments: Frank, Emilsson, Peloquin, Goodman &
Potts, "Rational speech act models of pragmatic reasoning in reference games"
(the PI's lab; reuse approved by the PI). Copied from
github.com/langcog/pragmods_expts at commit
`8ff10c31e47e0daf00c1f48ded583b955eb56747` (2016-09-19), directory `images/`
(the repository carries no licence file).

A referent is one base image (`<item>-base1..3.png`) with the overlay of each
of its features stacked on top (all 300x300 RGBA, transparent outside the
drawn part), as pragmods' `stimHTML` drew them. Only the bases and feature
overlays the experiment uses are copied: 6 items x 3 bases + 22 overlays = 40
files, ~190 KB.

| item | features (overlay file suffix) |
|---|---|
| friend | hat, glasses, mustache, bowtie |
| snowman | hat, scarf, mittens, belt |
| sundae | cherry, whipped_cream, chocolate, banana, sprinkles |
| pizza | mushrooms, olives, peppers |
| boat | cabin, sail, motor |
| christmas_tree | lights, ornaments, star |

Changes from the originals:

* file names are lower-cased with spaces replaced by underscores
  (`Christmas tree-star.png` -> `christmas_tree-star.png`,
  `sundae-whipped cream.png` -> `sundae-whipped_cream.png`);
* the five boat images that were 1250x1250 (`boat-base1..3`, `boat-cabin`,
  `boat-sail`) are downscaled to 300x300 (Lanczos), like every other image;
* every other file is byte-identical to the original.

The words Bob says and the alt-text phrases for each feature are in
`DOMAINS` in `src/rsa/experiment/design.py`.

## `demo_design.json`

Eight pragmods contexts for a demonstration (labels follow
`../data/README.md`): the simple game at levels 1 and 0, the complex game at
levels 1 and 2 and its prior (mumble) query, the twins display with the word
true of all three objects, the odd-one-out display, and the 4-object x
4-feature size display `sl20`.

```bash
uv run python -m src.rsa.experiment.design --design <design.json> --out lists.json --seed 1 --n-lists 40
uv run python -m src.rsa.experiment.build --trial-lists lists.json --out-dir site/
uv run python -m src.rsa.experiment.build --design src/pipelines/outer_loop/projects/rsa_reference/experiment/demo_design.json \
    --seed 7 --n-lists 1 --single-file preview.html --preview
uv run python -m src.rsa.experiment.convert --data data.json --participant-id p1 --experiment rsa_e1 --out rows.csv
```
