"""Each earlier paradigm's best theory stays carried as the 4-SE guard's reference (PI
2026-10-26; src/rsa_open/loop/orchestrator.py `OpenLoop.references`). Run B′ pruned E's best
theory on the newest block alone, and the guard then compared only survivors."""

from types import SimpleNamespace

import numpy as np

from src.rsa_open.corpus import CorpusCV


def _loop(tmp_path):
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    rng = np.random.default_rng(0)
    n = 40
    loop = OpenLoop.__new__(OpenLoop)
    loop.cfg = OpenLoopConfig(corpus_path=tmp_path, seed_models_dir=tmp_path, results_dir=tmp_path, max_iterations=1)
    loop.dir = tmp_path
    loop.models_dir = tmp_path / "models"
    loop.models_dir.mkdir()
    loop.blocks = [SimpleNamespace(name="E", units=tuple(range(n)), n=n),
                   SimpleNamespace(name="exp1", units=tuple(range(n)), n=n)]
    noise = lambda: rng.normal(0, 0.5, n)  # noqa: E731
    pointwise = {  # (E, exp1) per-trial held-out lpd
        "e_best": (np.zeros(n), np.full(n, -2.0)),           # best on E, far behind on exp1
        "incumbent": (-0.1 + noise(), np.zeros(n)),           # best on exp1, within the guard on E
        "weak": (-1.0 + noise(), np.full(n, -3.0)),           # behind on both
    }
    loop.cv = {k: CorpusCV({"E": e, "exp1": x}, True) for k, (e, x) in pointwise.items()}
    loop.cv_problems, loop.admission_limit_sec = {}, None
    loop.live_cfg = SimpleNamespace(time_limit_sec=None)
    loop.live = {}
    for name in pointwise:
        (loop.models_dir / f"{name}.py").write_text("# theory\n")
        loop.live[name] = SimpleNamespace(name=name, hypothesis=name, source="seed",
                                          fit=SimpleNamespace(converged=True, timing={}))
    loop.ledger = HypothesisLedger.create(tmp_path / "attempted_hypotheses.jsonl", inherit_from=None)
    loop.history, loop.step = [], 0
    loop._write_manifest = lambda: None
    return loop


def test_an_earlier_paradigm_s_best_theory_is_kept_though_far_behind_on_the_newest(tmp_path):
    loop = _loop(tmp_path)
    entry = loop.end()
    assert entry["best_model"] == "incumbent"
    assert set(loop.live) == {"incumbent", "e_best"}  # weak pruned; E's best kept
    kept = [e for e in entry["events"] if e["outcome"] == "kept"]
    assert kept == [dict(name="e_best", outcome="kept", reason="the best theory on E, the guard's reference there")]
    assert (tmp_path / "models" / "pruned" / "weak.py").exists()


def test_the_cap_never_retires_a_reference(tmp_path):
    loop = _loop(tmp_path)
    loop.cfg.max_live_models = 1
    loop.cfg.prune_dse_multiplier = 1e9  # nothing pruned on the newest block
    loop.cfg.guard_k = 1e9  # nor by the guard
    entry = loop.end()
    assert set(loop.live) == {"incumbent", "e_best"}  # over the cap: the reference stays
    assert any(e["outcome"] == "over cap" for e in entry["events"])
