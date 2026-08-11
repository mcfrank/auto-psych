"""A malformed model registry must fail loudly, not be silently coerced.

``load_registry`` already raises on unparseable YAML (a corrupt registry is not
an empty one). The same reasoning applies one level down: a ``theories`` block
that is not a mapping, or a non-numeric ``reserved_for_new``, used to be
replaced by ``{}`` / the default. That silently discards every accumulated
theory probability and hands the designer a uniform prior it never earned.
"""

from __future__ import annotations

import pytest

from src.registry.io import DEFAULT_RESERVED_FOR_NEW, load_registry, normalize_theories


def _write(tmp_path, text: str):
    path = tmp_path / "model_registry.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_registry_is_an_empty_registry(tmp_path):
    reg = load_registry(tmp_path / "absent.yaml")
    assert reg == {"theories": {}, "reserved_for_new": DEFAULT_RESERVED_FOR_NEW}


def test_well_formed_registry_round_trips(tmp_path):
    path = _write(tmp_path, "theories:\n  a: 0.5\n  b: 0.25\nreserved_for_new: 0.25\n")
    reg = load_registry(path)
    assert reg["theories"] == {"a": 0.5, "b": 0.25}
    assert reg["reserved_for_new"] == 0.25


def test_non_mapping_theories_raises(tmp_path):
    path = _write(tmp_path, "theories:\n  - a\n  - b\nreserved_for_new: 0.25\n")
    with pytest.raises(ValueError, match="theories"):
        load_registry(path)


def test_non_numeric_reserved_for_new_raises(tmp_path):
    path = _write(tmp_path, "theories:\n  a: 1.0\nreserved_for_new: lots\n")
    with pytest.raises(ValueError, match="reserved_for_new"):
        load_registry(path)


def test_non_mapping_registry_root_raises(tmp_path):
    path = _write(tmp_path, "- theories\n- reserved_for_new\n")
    with pytest.raises(ValueError, match="mapping"):
        load_registry(path)


@pytest.mark.parametrize("weight", [-0.1, ".nan", ".inf", "lots", True])
def test_invalid_theory_weight_raises(tmp_path, weight):
    rendered = str(weight).lower() if isinstance(weight, bool) else weight
    path = _write(tmp_path, f"theories:\n  a: {rendered}\nreserved_for_new: 0.25\n")
    with pytest.raises(ValueError, match="a"):
        load_registry(path)


@pytest.mark.parametrize("reserved", [-0.1, 1.1, ".nan", ".inf"])
def test_out_of_range_reserved_mass_raises(tmp_path, reserved):
    path = _write(
        tmp_path, f"theories:\n  a: 0.5\nreserved_for_new: {reserved}\n"
    )
    with pytest.raises(ValueError, match="reserved_for_new"):
        load_registry(path)


def test_normalize_rejects_negative_weight():
    with pytest.raises(ValueError, match="a"):
        normalize_theories({"a": -1.0, "b": 2.0})
