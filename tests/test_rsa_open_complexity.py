"""Code-size complexity (src/rsa_open/complexity.py): what it reads, and that it orders the standard-RSA grid, whose
complexity is known by construction."""

import itertools

from src.rsa_open.complexity import MEASURES, code_only, measure

THEORY = '''"""A long hypothesis.

It costs nothing."""

import jax.numpy as jnp

PARAMS = {"alpha": dist.LogNormal(0.0, 1.0)}  # a comment


def listener(params, ctx):
    """Docstring."""
    # a comment line
    return jnp.ones(3) * params["alpha"]
'''


def test_comments_docstrings_and_blank_lines_cost_nothing():
    code = code_only(THEORY)
    assert "hypothesis" not in code and "comment" not in code and "Docstring" not in code
    assert measure(THEORY) == measure(THEORY.replace('"""A long hypothesis.\n\nIt costs nothing."""', '"""Short."""'))
    m = measure(THEORY)
    assert m["lines"] == 4 and m["n_params"] == 1 and m["hand_set_numbers"] == 0


def test_hand_set_numbers_are_counted():
    assert measure(THEORY.replace('params["alpha"]', 'params["alpha"] * 2.5'))["hand_set_numbers"] == 1


def test_every_measure_but_lines_rises_when_a_grid_axis_is_switched_on():
    from src.rsa_open.grid import all_points, source

    points = {p.name: p for p in all_points()}
    m = {n: measure(source(p)) for n, p in points.items()}
    rises = {k: [0, 0] for k in MEASURES}
    for a, b in itertools.permutations(points.values(), 2):
        changed = [ax for ax in ("depth", "salience", "kind", "semantics", "cost") if getattr(a, ax) != getattr(b, ax)]
        added = ((changed == ["depth"] and b.depth == a.depth + 1)
                 or (set(changed) <= {"salience", "kind"} and a.salience == "none" and b.salience != "none")
                 or (changed == ["semantics"] and b.semantics == "graded")
                 or (changed == ["cost"] and b.cost != "none"))
        if added:
            for k in MEASURES:
                rises[k][0] += m[b.name][k] > m[a.name][k]
                rises[k][1] += 1
    for k in ("ast_nodes", "halstead_volume", "compressed_bytes"):
        assert rises[k][0] == rises[k][1], (k, rises[k])
    assert rises["lines"][0] < rises["lines"][1]  # lines miss some (graded meanings change a line, add none)
