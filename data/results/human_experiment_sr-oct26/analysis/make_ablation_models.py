"""Write the person-level ablation models: structure x personalisation.

Two mechanical transformations, so every variant differs from its source in
the person-level parameters alone:

* **Personalised seeds** (``<seed>__personal``): each seed's choice rule
  ``sigmoid(beta * (score_a - score_b) + side_bias)`` gets a person-level
  sensitivity ``beta_i`` (log-normal population, non-centred) and a
  person-level lapse ``lapse_i`` (logit-normal population, the winners'
  priors): ``p_left = 0.5 lapse_i + (1 - lapse_i) sigmoid(beta_i (...) + side_bias)``.
  ``side_bias`` and the seed's own parameters stay shared.
* **Pooled winners** (``<winner>__pooled``, ``__lapse_only``,
  ``__sens_lapse``): every person-level parameter of a winner is the
  non-centred ``mu + sigma * z[participant]``; setting ``sigma`` to 0 and
  dropping ``z`` gives everyone the population value. ``__lapse_only`` keeps
  the person-level lapse, ``__sens_lapse`` the lapse and the sensitivity
  (``beta``) — the same two the personalised seeds get. Run 2's winners have
  no lapse: ``__sens`` keeps their person-level sensitivity alone.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/make_ablation_models.py [--run 2]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")


def run_root(run: int) -> Path:
    return here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"


def ablation_dir(run: int) -> Path:
    return ANALYSIS_ROOT / ("ablation_models" if run == 1 else f"ablation_models_run{run}")


# Per run: each experiment's winner -> the experiment whose model_loop/models/ holds it.
WINNERS = {
    1: {
        "individual_streak_aversion_lapse": 1,
        "most_lopsided_stretch_aversion": 2,
        "personal_pattern_detection_gain": 3,
    },
    2: {
        "graded_periodicity_personal_ideal": 1,
        "tally_span_switch_ideal_periodic_unit": 2,
        "edge_streak_primacy_recency_pattern_rule": 3,
    },
}
# Which person-level parameters each pooled variant keeps (by the suffix of
# its z_ vector's name).
VARIANTS = {
    1: {"pooled": (), "lapse_only": ("lapse",), "sens_lapse": ("lapse", "beta")},
    2: {"pooled": (), "sens": ("beta",)},
}

SEED_CHOICE = re.compile(
    r"pm\.math\.sigmoid\(beta \* \((?P<diff>[^\n]+?)\) \+ side_bias\)"
)
PERSONAL_PRIORS = """
    # Person-level sensitivity and lapse (ablation: personalised seed).
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    mu_beta_p = pm.Normal("mu_beta_p", mu=1.0, sigma=1.0)
    sigma_beta_p = pm.HalfNormal("sigma_beta_p", sigma=0.7)
    z_beta_p = pm.Normal("z_beta_p", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta_p = pm.math.exp(mu_beta_p + sigma_beta_p * z_beta_p)
    mu_lapse_p = pm.Normal("mu_lapse_p", mu=-2.0, sigma=1.0)
    sigma_lapse_p = pm.HalfNormal("sigma_lapse_p", sigma=1.0)
    z_lapse_p = pm.Normal("z_lapse_p", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse_p = pm.math.sigmoid(mu_lapse_p + sigma_lapse_p * z_lapse_p)
"""


@dataclass
class Args:
    """Write the ablation model files and their manifest."""

    run: int = 1
    out_dir: Path | None = None
    """Default: ``ablation_models`` (run 1), ``ablation_models_run<N>``."""


def personalise_seed(source: str, name: str) -> str:
    if len(SEED_CHOICE.findall(source)) != 1:
        raise ValueError(f"{name}: expected one sigmoid(beta * (...) + side_bias) choice rule")
    # The shared beta is replaced, not kept beside the personal one.
    source, n = re.subn(r'\n    beta = pm\.Uniform\("beta", lower=0\.2, upper=12\.0\)', "", source)
    if n != 1:
        raise ValueError(f"{name}: expected one shared beta prior")
    source, n = re.subn(
        r'(\n    side_bias = pm\.Uniform\("side_bias", lower=-2\.0, upper=2\.0\)\n)',
        lambda m: m[1] + PERSONAL_PRIORS,
        source,
    )
    if n != 1:
        raise ValueError(f"{name}: expected one side_bias prior")
    source = SEED_CHOICE.sub(
        lambda m: (
            "pt.clip(0.5 * lapse_p[participant_id] + (1.0 - lapse_p[participant_id])"
            f" * pm.math.sigmoid(beta_p[participant_id] * ({m['diff']}) + side_bias), 1e-6, 1 - 1e-6)"
        ),
        source,
    )
    if "import pytensor.tensor as pt" not in source:
        source = source.replace("import pymc as pm\n", "import pymc as pm\nimport pytensor.tensor as pt\n", 1)
    source = source.replace("\nwith pm.Model() as model:", "\nMAX_PARTICIPANTS = 400\n\nwith pm.Model() as model:", 1)
    if name == "motif_stack":
        # Its prepare_observed hook builds every pm.Data array itself.
        old = '        "chose_left": chose_left,\n    }'
        if old not in source:
            raise ValueError("motif_stack: prepare_observed return not found")
        # Like chose_left: every row's id, or zeros for stimulus-only rows
        # (its placeholder and prediction pools); never some of each.
        source = source.replace(
            "    return {\n        **_unique_sequence_table(sequences),",
            '    has_id = ["participant_id" in row for row in rows]\n'
            "    if any(has_id) and not all(has_id):\n"
            '        raise ValueError("participant_id is present on some rows but not others.")\n'
            "    participant_id = np.array(\n"
            '        [int(float(row["participant_id"])) for row in rows] if all(has_id) else [0] * len(rows),\n'
            '        dtype="int64",\n'
            "    )\n\n"
            "    return {\n        **_unique_sequence_table(sequences),",
            1,
        )
        source = source.replace(
            old, '        "chose_left": chose_left,\n        "participant_id": participant_id,\n    }', 1
        )
    return f'"""Ablation: {name} with a person-level sensitivity and lapse."""\n' + source


def pool_winner(source: str, name: str, keep: tuple[str, ...]) -> str:
    # Each person-level vector z_<x> (one entry per participant slot) and the
    # sigma it is scaled by (``sigma_<y> * z_<x>``; run 2 has sigma_log_beta * z_beta).
    zs = dict(
        re.findall(r'\n    z_(\w+) = pm\.Normal\("z_\w+", mu=0\.0, sigma=1\.0, shape=(MAX_PARTICIPANTS|N_SLOTS)\)', source)
    )
    sigmas = {}
    for suffix in zs:
        scaled_by = set(re.findall(rf"\b(sigma_\w+) \* z_{suffix}\b", source))
        if len(scaled_by) != 1 or not re.search(
            rf'\n    {next(iter(scaled_by))} = pm\.HalfNormal\("{next(iter(scaled_by))}", sigma=[\d.]+\)', source
        ):
            raise ValueError(f"{name}: z_{suffix} is not scaled by one HalfNormal sigma (found {scaled_by})")
        sigmas[suffix] = next(iter(scaled_by))
    if not zs:
        raise ValueError(f"{name}: no person-level z vectors")
    unknown = set(keep) - set(zs)
    if unknown:
        raise ValueError(f"{name}: no person-level {unknown} to keep (has {sorted(zs)})")
    for suffix, sigma in sigmas.items():
        if suffix in keep:
            continue
        source, n = re.subn(
            rf'\n    {sigma} = pm\.HalfNormal\("{sigma}", sigma=[\d.]+\)', f"\n    {sigma} = 0.0", source
        )
        source, m = re.subn(
            rf'\n    z_{suffix} = pm\.Normal\("z_{suffix}", mu=0\.0, sigma=1\.0, shape={zs[suffix]}\)',
            f"\n    z_{suffix} = pt.zeros({zs[suffix]})",
            source,
        )
        if (n, m) != (1, 1):
            raise ValueError(f"{name}: could not pool {sigma} / z_{suffix}")
    if "import pytensor.tensor as pt" not in source:
        source = source.replace("import pymc as pm\n", "import pymc as pm\nimport pytensor.tensor as pt\n", 1)
    kept = ", ".join(keep) or "none"
    return f'"""Ablation: {name} with person-level parameters pooled (kept: {kept})."""\n' + source


def main(args: Args) -> None:
    root = run_root(args.run)
    out_dir = args.out_dir or ablation_dir(args.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for seed in SEEDS:
        src = (root / "experiment1/cognitive_models_input" / f"{seed}.py").read_text(encoding="utf-8")
        name = f"{seed}__personal"
        (out_dir / f"{name}.py").write_text(personalise_seed(src, seed), encoding="utf-8")
        names.append(name)
    for winner, experiment in WINNERS[args.run].items():
        src = (root / f"experiment{experiment}/model_loop/models/{winner}.py").read_text(encoding="utf-8")
        for variant, keep in VARIANTS[args.run].items():
            if "beta" in keep and "z_beta" not in src:
                continue  # the winner's sensitivity is shared already: sens_lapse == lapse_only
            name = f"{winner}__{variant}"
            (out_dir / f"{name}.py").write_text(pool_winner(src, winner, keep), encoding="utf-8")
            names.append(name)
    manifest = "models:\n" + "".join(f"  - name: {n}\n" for n in names)
    (out_dir / "models_manifest.yaml").write_text(manifest, encoding="utf-8")
    print("\n".join(names))


if __name__ == "__main__":
    main(tyro.cli(Args))
