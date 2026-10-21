"""The rehearsal's ground truth (src/rsa_open/ground_truth.py): ported, refused when the chain
starts with it, fitted once to the literature and fixed there."""

import json
import shutil

import numpy as np
import pytest
import yaml

from src.rsa.fit import FitSettings
from tests.test_rsa_open_design import SEEDS
from tests.test_rsa_open_loop_orchestrator import ported

SETTINGS = FitSettings(num_warmup=150, num_samples=150, num_chains=2, seed=0, dense_mass=True)


@pytest.fixture(scope="module")
def setting(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from tests.test_rsa_open_corpus import simulated_block

    root = tmp_path_factory.mktemp("gt")
    bar = root / "bar"
    bar.mkdir()
    for name in ("literal_listener", "rsa_l1"):
        (bar / f"{name}.py").write_text(ported(name))
    (bar / "models_manifest.yaml").write_text(yaml.safe_dump(
        {"models": [{"name": n, "rationale": "x"} for n in ("literal_listener", "rsa_l1")]}))
    (root / "truth.py").write_text(ported("rsa_l1_salience"))
    literature = simulated_block(TheoryModel(root / "truth.py"), "E", likelihood("choice"), 120, 1,
                                 {"temperature": 1.0, "lapse": 0.05})
    return root, bar, literature


def test_a_theory_the_chain_starts_with_is_refused(setting, tmp_path):
    from src.rsa_open.ground_truth import GroundTruthRefused, make_ground_truth

    root, bar, literature = setting
    kw = dict(bar=bar, cache=tmp_path / "cache", settings=SETTINGS, literature=literature)
    with pytest.raises(GroundTruthRefused, match="the bar has a theory named rsa_l1"):
        make_ground_truth(SEEDS / "rsa_l1.py", out=tmp_path / "a", **kw)
    shutil.copyfile(SEEDS / "rsa_l1.py", tmp_path / "renamed_rsa.py")
    with pytest.raises(GroundTruthRefused, match="its port is the bar's rsa_l1"):
        make_ground_truth(tmp_path / "renamed_rsa.py", out=tmp_path / "b", **kw)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


@pytest.mark.slow  # one fit to the literature block
def test_the_truth_is_fixed_at_its_posterior_means_on_the_literature(setting, tmp_path):
    from src.rsa_open.chain import GroundTruth
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.ground_truth import RECORD_FILE, load_ground_truth, make_ground_truth
    from src.rsa_open.links import LINKS

    root, bar, literature = setting
    out = tmp_path / "private"
    record = make_ground_truth(SEEDS / "rsa_l1_salience.py", bar=bar, out=out, cache=tmp_path / "cache",
                               settings=SETTINGS, literature=literature, provenance={"why": "test"})
    gt = load_ground_truth(out)
    assert isinstance(gt, GroundTruth) and set(gt.theory_params) == set(TheoryModel(gt.theory_path).params)
    assert set(gt.link_params) == set(LINKS)
    assert all(set(p) == {"temperature", "lapse"} for p in gt.link_params.values())
    assert len({json.dumps(p, sort_keys=True) for p in gt.link_params.values()}) == 1  # E's, for every link
    assert 0 < gt.link_params["choice"]["lapse"] < 0.5
    assert record["ground_truth"] == "rsa_l1_salience" and record["provenance"] == {"why": "test"}
    assert set(record["fit"]["posterior_sd"]) >= {"temperature@E", "lapse@E"}
    assert np.isfinite(list(gt.theory_params.values())).all()
    # Once: the same model returns its record; another model is refused.
    assert make_ground_truth(SEEDS / "rsa_l1_salience.py", bar=bar, out=out, cache=tmp_path / "cache",
                             settings=SETTINGS, literature=literature) == record
    with pytest.raises(FileExistsError):
        make_ground_truth(SEEDS / "literal_listener.py", bar=root, out=out, cache=tmp_path / "cache",
                          settings=SETTINGS, literature=literature)
    # A changed theory file is caught on load.
    gt.theory_path.write_text(gt.theory_path.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="is not the theory"):
        load_ground_truth(out)
    assert (out / RECORD_FILE).exists()


def test_the_committed_seeds_are_the_closed_campaign_s_ported(tmp_path):
    """data/rsa_open/seeds is `src.rsa_open.seeds.build`'s output, byte for byte."""
    from src.rsa_open.seeds import OUT, build

    record = build(tmp_path)
    assert len(record["bar"]) == 15 and set(record["chains"]) == {"chain_0", "chain_1", "chain_2"}
    assert all({"rsa_l1", "rsa_l2"} <= set(m) and set(m) <= set(record["bar"]) for m in record["chains"].values())
    built = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    committed = sorted(p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file())
    assert built == committed
    assert all((tmp_path / p).read_bytes() == (OUT / p).read_bytes() for p in built)


def test_no_file_under_src_names_the_rehearsal_truth():
    """Theorists read src/ (CAMPAIGN_LOG #6: agents never see a simulated run's ground truth)."""
    from pyprojroot import here

    from src.rsa_open.ground_truth import rehearsal_truth

    name = rehearsal_truth().stem
    assert not [p for p in (here() / "src").rglob("*") if p.is_file() and p.suffix != ".pyc"
                and name in p.read_text(errors="ignore")]
