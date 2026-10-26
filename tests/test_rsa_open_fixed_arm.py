"""The fixed arm (PI 2026-10-26): the closed campaign's forced-choice paradigm as one frozen spec
(src/rsa_open/fixed.py), its pragmods pictures as trusted images with base variants, and the
chain mode that sends it through the harness's own checks with no designer."""

import base64
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_rsa_open_paradigm import closed_like_spec, write

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_spec_is_the_builder_s_and_valid(tmp_path):
    from src.rsa_open.fixed import SPEC_PATH, build, write as write_spec
    from src.rsa_open.paradigm import BUDGET_USD, load_paradigm

    assert json.loads(SPEC_PATH.read_text()) == build()
    p = load_paradigm(write_spec(tmp_path / "fixed.json"))
    assert p.harness_materials and p.response.type == "choice"
    assert [d.name for d in p.materials.domains] == ["friend", "snowman", "sundae", "pizza", "boat", "Christmas tree"]
    assert all(len(d.variants) == 3 for d in p.materials.domains)
    assert p.design.n_participants == 400 and p.cost_usd == pytest.approx(BUDGET_USD, abs=0.01)
    assert "Bob can only say one word" in p.response.prompt and "mumbles" in p.response.prior_prompt


def _png_image(data: bytes) -> str:
    return f'<image href="data:image/png;base64,{base64.b64encode(data).decode()}" width="100" height="100"/>'


def test_only_the_curated_pictures_may_be_embedded(tmp_path):
    from src.rsa_open.paradigm import TRUSTED_IMAGES_DIR, ParadigmInvalid, load_paradigm

    trusted = (TRUSTED_IMAGES_DIR / "friend-base1.png").read_bytes()
    spec = closed_like_spec()
    spec["materials"]["domains"][0]["base"] = _png_image(trusted)
    assert load_paradigm(write(tmp_path, spec)).harness_materials
    for href in (_png_image(trusted[:-10] + b"0123456789"),  # any other bytes
                 '<image href="https://example.org/x.png" width="100" height="100"/>',
                 '<image href="data:image/svg+xml;base64,PHN2Zy8+" width="100" height="100"/>'):
        other = copy.deepcopy(spec)
        other["materials"]["domains"][0]["base"] = href
        with pytest.raises(ParadigmInvalid, match="image"):
            load_paradigm(write(tmp_path, other, name="other.json"))


def test_each_object_gets_a_different_base_variant_up_to_their_number():
    from src.rsa_open.fixed import SPEC_PATH
    from src.rsa_open.paradigm import load_paradigm
    from src.rsa_open.screens import trial_lists
    from tests.test_rsa_open_screens import units_of

    p = load_paradigm(SPEC_PATH)
    lists = trial_lists(p, units_of(p), seed=0, n_lists=3)
    for trial in (t for lst in lists["lists"] for t in lst["trials"]):
        v = trial["variants"]
        assert len(v) == len(trial["objects"]) and len(set(v)) == min(3, len(v))
        assert all(p.materials.domains[trial["domain"]].variants[k][:60] in html
                   for k, html in zip(v, trial["items_html"]))
        assert len(p.materials.domains[trial["domain"]].features) >= len(trial["objects"][0])


def test_a_paradigm_without_variants_draws_the_same_lists_as_before():
    from src.rsa_open.paradigm import parse_paradigm
    from src.rsa_open.screens import trial_lists
    from tests.test_rsa_open_screens import units_of

    p = parse_paradigm(closed_like_spec())
    lists = trial_lists(p, units_of(p), seed=0, n_lists=2)
    assert all("variants" not in t for lst in lists["lists"] for t in lst["trials"])


def test_a_domain_needs_only_the_features_of_the_scenes_it_can_draw(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()  # scenes of 2-4 features; the one domain has 4
    three = copy.deepcopy(spec["materials"]["domains"][0])
    three.update(name="hats", features=three["features"][:3])
    spec["materials"]["domains"].append(three)
    load_paradigm(write(tmp_path, spec))  # a 3-feature domain serves the 2-3-feature scenes
    spec["materials"]["domains"] = [three]
    with pytest.raises(ParadigmInvalid, match="no domain names 4 features"):
        load_paradigm(write(tmp_path, spec, name="narrow.json"))
    one = copy.deepcopy(three)
    one.update(name="caps", features=three["features"][:1])
    spec["materials"]["domains"] = [closed_like_spec()["materials"]["domains"][0], one]
    with pytest.raises(ParadigmInvalid, match="every scene needs at least 2"):
        load_paradigm(write(tmp_path, spec, name="one.json"))


def test_the_harness_s_pictures_skip_the_legibility_check():
    from src.rsa_open.fixed import SPEC_PATH
    from src.rsa_open.paradigm import load_paradigm
    from src.rsa_open.pilot import materials_check

    out = materials_check(load_paradigm(SPEC_PATH), checkers=[])
    assert out["items"] == 0 and out["legibility"] is None


def test_the_fixed_arm_submits_the_spec_to_the_harness_s_checks_salted_by_chain_and_experiment(tmp_path, monkeypatch):
    import src.rsa_open.designer as designer
    from src.rsa_open.chain import Chain, ChainConfig, FixedDesignFailed
    from src.rsa_open.designer import SubmissionResult

    calls = []

    def fake_check(ws, submitted, out, *, pilot_participants, participant_factory=None, salt=""):
        calls.append((json.loads(Path(submitted).read_text()), salt))
        return SubmissionResult("x", ok, [] if ok else ["pilot: too slow"], out, {})

    monkeypatch.setattr(designer, "check_submission", fake_check)
    monkeypatch.setattr(designer, "prepare_workspace", lambda root, **kw: __import__("src.rsa_open.tools", fromlist=["x"])
                        .Workspace(Path(root)) if not (Path(root) / "work").mkdir(parents=True) else None)
    monkeypatch.setattr(designer, "freeze", lambda result, out, **kw: (out.mkdir(parents=True), kw)[1])
    cfg = ChainConfig(chain="chain_3", arm="fixed", seeds_dir=tmp_path, literature=None, out_dir=tmp_path / "run",
                      ground_truth=None)
    c = Chain(cfg, designer_spawn=None, theorist_spawn=None)
    monkeypatch.setattr(c, "design_context", lambda n: {})
    monkeypatch.setattr(c, "history", lambda n: [])
    monkeypatch.setattr(c, "timing", lambda n: [])
    ok = True
    assert c.design(1) == tmp_path / "run" / "experiment1" / "design" / "frozen"
    record, salt = calls[-1]
    assert record["submitted_by"] == "fixed arm" and Path(record["spec"]).name == "paradigm.json"
    assert salt == "chain_3/experiment1"
    ok = False
    with pytest.raises(FixedDesignFailed, match="pilot: too slow"):
        c.design(2)
    assert calls[-1][1] == "chain_3/experiment2"


def test_the_open_arm_needs_a_designer_and_arms_are_named(tmp_path):
    from src.rsa_open.chain import Chain, ChainConfig

    open_chain = Chain(ChainConfig(chain="c", seeds_dir=tmp_path, literature=None, out_dir=tmp_path / "o",
                                   ground_truth=None), designer_spawn=None, theorist_spawn=None)
    open_chain.design_context = lambda n: {}
    with pytest.raises(ValueError, match="needs a designer"):
        open_chain.design(1)
    with pytest.raises(ValueError, match="arm 'closed'"):
        Chain(ChainConfig(chain="c", arm="closed", seeds_dir=tmp_path, literature=None, out_dir=tmp_path,
                          ground_truth=None), designer_spawn=None, theorist_spawn=None)


def test_the_fixed_arm_s_files_stay_out_of_the_agents_tree():
    exclude = (ROOT / "scripts/rsa/slurm/agent_tree.exclude").read_text().split()
    assert "/src/rsa_open/fixed.py" in exclude and "/src/rsa_open/fixed_paradigm.json" in exclude
    text = (ROOT / "scripts/rsa/slurm/open_rehearsal.sbatch").read_text()
    assert 'CELL="${ARM}${SPEC_LABEL}_chain_$CHAIN"' in text and '--arm "$ARM"' in text


@pytest.mark.slow  # the design fits, a real EIG pick and a scripted pilot of the fixed spec
def test_a_fixed_experiment_is_designed_end_to_end(tmp_path, monkeypatch):
    import src.rsa_open.design as design
    import src.rsa_open.pilot as pilot
    import src.rsa_open.tools as tools
    from src.rsa_open.chain import Chain, ChainConfig, GroundTruth
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from tests.test_rsa_open_chain import SETTINGS
    from tests.test_rsa_open_loop_orchestrator import ported
    from tests.test_rsa_open_corpus import simulated_block
    from tests.test_rsa_open_pilot import Scripted
    import yaml

    monkeypatch.setattr(design, "DEFAULT_DRAWS", 16)
    monkeypatch.setattr(design, "DEFAULT_SCENARIOS", 100)
    monkeypatch.setattr(tools, "link_check_report", lambda ws, spec, **kw: {
        "paradigm_sha256": __import__("src.rsa_open.paradigm", fromlist=["x"]).load_paradigm(spec).sha256,
        "link": "choice", "theory": "x", "n_trials": 1, "converged": True, "recovered": True, "misses": {}})
    real_pilot = pilot.run_pilot
    monkeypatch.setattr(pilot, "run_pilot", lambda paradigm, lists, out, *, n_participants, participant_factory=None,
                        max_workers=6, timing=None: real_pilot(
                            paradigm, lists, out, n_participants=n_participants, max_workers=1, timing=timing,
                            participant_factory=lambda i: Scripted(lists, i, paradigm=paradigm)))
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for name in ("literal_listener", "rsa_l1"):
        (seeds / f"{name}.py").write_text(ported(name))
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "literal_listener"},
                                                                            {"name": "rsa_l1"}]}))
    gt_path = tmp_path / "gt.py"
    gt_path.write_text(ported("rsa_l1_salience"))
    literature = simulated_block(TheoryModel(gt_path), "E", likelihood("choice"), 120, 1,
                                 {"temperature": 1.0, "lapse": 0.05})
    cfg = ChainConfig(chain="chain_0", arm="fixed", seeds_dir=seeds, literature=literature, out_dir=tmp_path / "run",
                      ground_truth=GroundTruth(gt_path, {"alpha": 2.0, "w_features": 0.5, "w_familiar": 0.0}),
                      settings=SETTINGS, design_settings=SETTINGS, pilot_participants=3)
    frozen = Chain(cfg, designer_spawn=None, theorist_spawn=None).design(1)
    record = json.loads((frozen / "record.json").read_text())
    assert record["submitted_by"] == "fixed arm" and record["experiment"] == 1
    lists = json.loads((frozen / "lists.json").read_text())
    assert len(lists["lists"]) == 400  # one list per person
    assert all("variants" in t for t in lists["lists"][0]["trials"])
    assert np.isfinite(record["bits_per_dollar"])


def test_the_two_measure_rehearsal_variant_is_the_builder_s_valid_and_within_budget():
    from src.rsa_open.fixed import PAIR_SPEC_PATH, build_pair
    from src.rsa_open.paradigm import BUDGET_USD, load_paradigm

    assert json.loads(PAIR_SPEC_PATH.read_text()) == build_pair()
    p = load_paradigm(PAIR_SPEC_PATH)
    assert p.second_response.type == "rating" and p.cost_usd <= BUDGET_USD
    assert (p.design.n_participants * p.design.trials_per_participant) % p.design.distinct_trials == 0
    exclude = (ROOT / "scripts/rsa/slurm/agent_tree.exclude").read_text().split()
    assert "/src/rsa_open/fixed_pair_paradigm.json" in exclude
    text = (ROOT / "scripts/rsa/slurm/open_rehearsal.sbatch").read_text()
    assert 'CELL="${ARM}${SPEC_LABEL}_chain_$CHAIN"' in text and "--fixed-spec" in text
