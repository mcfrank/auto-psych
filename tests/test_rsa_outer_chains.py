"""Splitting the promoted seeds across the live chains (PI 2026-10-09).

Each chain gets the reference model and a share of the others, chosen so that
the models within a chain disagree as much as possible on the displays the
live experiments can show: near-twins go to different chains.
"""

import numpy as np
import pytest
import yaml

from src.rsa.outer.chains import split, write_chains


def preds_with_twins():
    # Three twins (a1-a3) that agree on the design pool, and six distinct models.
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(4), size=30)
    preds = {f"a{i}": base + 0.001 * i for i in (1, 2, 3)}
    for i in range(6):
        preds[f"m{i}"] = rng.dirichlet(np.ones(4), size=30)
    preds["ref"] = rng.dirichlet(np.ones(4), size=30)
    return preds


def test_twins_go_to_different_chains_and_the_reference_to_every_chain():
    chains = split(preds_with_twins(), n_chains=3, reference="ref")
    assert len(chains) == 3
    assert all(c[-1] == "ref" for c in chains)
    members = [c[:-1] for c in chains]
    assert sorted(len(m) for m in members) == [3, 3, 3]
    assert sorted(n for m in members for n in m) == sorted(f"a{i}" for i in (1, 2, 3)) + [f"m{i}" for i in range(6)]
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i < j:
                assert not any(f"a{i}" in m and f"a{j}" in m for m in members)


def test_the_split_is_deterministic_and_sizes_differ_by_at_most_one():
    preds = preds_with_twins()
    preds["m6"] = np.full((30, 4), 0.25)
    a = split(preds, n_chains=3, reference="ref")
    assert a == split(dict(reversed(list(preds.items()))), n_chains=3, reference="ref")
    assert sorted(len(c) - 1 for c in a) == [3, 3, 4]


def test_too_few_models_for_the_chains_raises():
    preds = {k: v for k, v in preds_with_twins().items() if k in ("a1", "a2", "ref")}
    with pytest.raises(ValueError, match="chains"):
        split(preds, n_chains=3, reference="ref")


def test_each_chain_is_a_seed_folder_with_its_manifest(tmp_path):
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    preds = preds_with_twins()
    entries = []
    for name in preds:
        (promoted / f"{name}.py").write_text(f"# {name}\n")
        entries.append(dict(name=name, rationale=f"why {name}", source="x"))
    (promoted / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}, sort_keys=False))
    chains = split(preds, n_chains=3, reference="ref")
    dirs = write_chains(promoted, chains, tmp_path / "chains")
    for d, chain in zip(dirs, chains):
        manifest = yaml.safe_load((d / "models_manifest.yaml").read_text())
        assert [e["name"] for e in manifest["models"]] == chain
        assert manifest["models"][0]["rationale"] == f"why {chain[0]}"
        assert sorted(p.stem for p in d.glob("*.py")) == sorted(chain)
