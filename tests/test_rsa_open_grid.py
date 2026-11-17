"""The factorial grid of standard RSA theories (src/rsa_open/grid.py, docs/auto_open/RSA_GRID.md): distinct points,
files that pass the admission gates, and corners that reproduce the existing standard theories exactly."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

BAR = Path("data/rsa_open/seeds/bar")


def _pool(n=40):
    from src.rsa.loop.novelty import plain_pool

    pool = plain_pool()
    return [pool[i] for i in np.linspace(0, len(pool) - 1, n).round().astype(int)]


def test_the_grid_has_36_distinct_points():
    from src.rsa_open.grid import all_points

    points = all_points()
    names = [p.name for p in points]
    assert len(points) == 36 and len(set(names)) == 36 and len(set(points)) == 36
    assert all(n.replace("_", "").isalnum() and n == n.lower() and len(n) <= 40 for n in names)
    assert sum(p.depth == 0 for p in points) == 6 and all(sum(p.depth == d for p in points) == 10 for d in (1, 2, 3))
    assert all(n.endswith("_nocost") for n in names)  # the suffix keeps fits made with the cost axis matched


def test_there_is_no_cost_axis():
    """Removed (PI 2026-11-17): with one-word utterances there is no production cost, and a free extension cost
    traded off with recursion depth."""
    from src.rsa_open.grid import GridPoint, all_points, source

    with pytest.raises(TypeError):
        GridPoint(1, cost="extension")
    assert not any("w_cost" in source(p) or "extension_share" in source(p) for p in all_points())


@pytest.mark.parametrize("bad", [dict(depth=4), dict(depth=1, salience="listener"), dict(depth=1, kind="fc"),
                                 dict(depth=0, salience="shared", kind="fc")])
def test_impossible_points_are_refused(bad):
    from src.rsa_open.grid import GridPoint

    with pytest.raises(ValueError):
        GridPoint(**bad)


def test_every_file_passes_the_admission_gates_that_read_the_source():
    from src.rsa.loop.code_gate import code_problems
    from src.rsa_open.grid import all_points, source
    from src.rsa_open.loop.gates import magic_number_problems, theory_shape_problems

    for p in all_points():
        text = source(p)
        assert not code_problems(text), (p.name, code_problems(text))
        assert not theory_shape_problems(text), (p.name, theory_shape_problems(text))
        assert not magic_number_problems(text), (p.name, magic_number_problems(text))


def _probs(path, params, contexts):
    from src.rsa.context import group_by_shape
    from src.rsa_open.fit import TheoryModel

    theory = TheoryModel(path)
    out = []
    for group in group_by_shape(contexts).values():
        out.append(np.asarray(theory.group_probs({k: jnp.asarray(v) for k, v in params.items()}, group)).ravel())
    return np.concatenate(out)


@pytest.mark.parametrize("point,reference,params,extra", [
    (dict(depth=0), "literal_listener", {}, {}),
    (dict(depth=1), "rsa_l1", {"alpha": 1.7}, {}),
    (dict(depth=2), "rsa_l2", {"alpha": 1.3}, {}),
    (dict(depth=1, salience="listener", kind="fc"), "rsa_l1_salience", {"alpha": 1.4, "w_salience": 0.6},
     {"w_features": 0.6, "w_familiar": 0.0}),
    (dict(depth=1, salience="shared", kind="fc"), "rsa_l1_shared_prior", {"alpha": 1.4, "w_salience": -0.5},
     {"w_features": -0.5, "w_familiar": 0.0}),
])
def test_the_corners_are_the_standard_theories(tmp_path, point, reference, params, extra):
    from src.rsa_open.grid import GridPoint, source

    p = GridPoint(**point)
    path = tmp_path / f"{p.name}.py"
    path.write_text(source(p))
    contexts = _pool()
    ref_params = {k: v for k, v in params.items() if k != "w_salience"} | extra
    np.testing.assert_allclose(_probs(path, params, contexts), _probs(BAR / f"{reference}.py", ref_params, contexts),
                               rtol=1e-5, atol=1e-6)


def test_each_axis_changes_the_predictions(tmp_path):
    from src.rsa_open.grid import GridPoint, source

    contexts = _pool()
    base = dict(depth=2, salience="shared", kind="fc", semantics="bool")
    params = {"alpha": 1.5, "w_salience": 0.7}

    def probs(**change):
        p = GridPoint(**(base | change))
        path = tmp_path / f"{p.name}.py"
        path.write_text(source(p))
        return _probs(path, {k: v for k, v in params.items() if k in p.params}, contexts)

    ref = probs()
    for change in (dict(depth=3), dict(salience="listener"), dict(kind="singleton"), dict(semantics="graded")):
        assert np.abs(probs(**change) - ref).max() > 1e-3, change


@pytest.mark.slow  # memo compiles every shape of every sampled point
def test_grid_theories_meet_the_listener_and_speaker_contracts(tmp_path):
    from src.rsa.context import group_by_shape
    from src.rsa.model_file import check_contract
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.grid import all_points, write_grid
    from src.rsa_open.loop.gates import check_speaker_contract

    points = all_points()
    sample = [p for p in points if p.depth == 0][:2] + [p for p in points if p.depth == 3]
    write_grid(tmp_path, sample)
    groups = group_by_shape(_pool(60))
    for p in sample:
        theory = TheoryModel(tmp_path / f"{p.name}.py")
        check_contract(theory, groups)
        check_speaker_contract(theory, groups)
