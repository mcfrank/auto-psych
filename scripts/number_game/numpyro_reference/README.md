# Eric Bigelow's numpyro number-game models (reference copy)

Verbatim copies of `src/prob_models/number_game/{infer,np_utils,run_number_eb}.py`
from `github.com/ebigelow/cognitive-mechanisms` at commit
`f27ec64d352ee4c66e8453a506cf144200386887` (2025-09-07, the last commit before
the directory was deleted in `1dc5cf7`). Not imported by the pipeline: the
PyMC seed ports in `src/pipelines/outer_loop/projects/number_game/seed_models/`
are checked against them by `scripts/number_game/check_pymc_port.py`, which
runs them under `uv run --with numpyro --with jax` (numpyro is not a project
dependency).

The model is `infer.number_bpl_model` with `prior_type='basic'`: 42 rule
concepts + 5,050 intervals (`np_utils.get_concept_sets`), p_mix ~ Beta(4, 4),
rule weights ~ Gamma(2, 1) normalised, intervals uniform, a strong-sampling
likelihood with epsilon = 0.1 uniform noise, membership-averaged predictions,
and a Binomial per (set, target) cell.
