"""A ``prepare_observed`` hook that needs a column the rows lack says so structurally.

``make_stim_data`` signals "these rows cannot bind this model" with
``MissingStimulusColumns``, which carries the missing names as data so callers
classify the failure without parsing a message — the novelty gate turns it into
a recorded rejection, the EIG screen into a recorded drop. The column-mapping
path always raised it; the hook path did not. A hook that reads
``row["participant_id"]`` off a bare stimulus row raised a raw ``KeyError``
instead, which no caller catches, and a GPT-6 Luna candidate that did exactly
that took down a whole holdout cell.

Only the response-row bookkeeping columns (``NON_STIMULUS_COLUMNS``) can be
legitimately absent from a stimulus row, so only a ``KeyError`` naming one of
those, *and* raised on rows that really lack it, is translated. Any other
``KeyError`` is a bug in the hook and propagates untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.data_binding import MissingStimulusColumns, make_stim_data
from src.models.model_loading import load_pymc_model

# The PyMC block every test model shares: one float input, the response.
_MODEL_BLOCK = '''

with pm.Model() as model:
    signal = pm.Data("signal", np.zeros(2))
    chose_left = pm.Data("chose_left", np.zeros(2, dtype="int64"))
    weight = pm.Normal("weight", 0.0, 1.0)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(weight * signal))
    pm.Bernoulli("obs", p=p_left, observed=chose_left)
'''


def _hook_model(tmp_path, name: str, signal_expression: str):
    """A model whose hook computes ``signal`` from ``signal_expression`` per row."""
    source = (
        "import numpy as np\nimport pymc as pm\n\n"
        "_TABLE = {'HHTT': 1.0}\n\n\n"
        "def prepare_observed(rows):\n"
        f"    signal = np.array([float({signal_expression}) for row in rows])\n"
        "    chose_left = np.array([int(row['chose_left']) for row in rows], dtype='int64')\n"
        "    return {'signal': signal, 'chose_left': chose_left}\n"
        + _MODEL_BLOCK
    )
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    return load_pymc_model(name, tmp_path)


STIMULUS_ROWS = [
    {"sequence_a": "HHTT", "sequence_b": "HTHT", "chose_left": 0},
    {"sequence_a": "HTTH", "sequence_b": "THTH", "chose_left": 0},
]


def test_hook_reading_participant_id_off_a_stimulus_row_is_a_missing_column(tmp_path):
    model = _hook_model(tmp_path, "reads_pid", "int(row['participant_id']) % 2")
    with pytest.raises(MissingStimulusColumns) as excinfo:
        make_stim_data(model, STIMULUS_ROWS)
    assert excinfo.value.missing == ("participant_id",)
    assert excinfo.value.only_non_stimulus


def test_hook_reading_trial_index_off_a_stimulus_row_is_a_missing_column(tmp_path):
    model = _hook_model(tmp_path, "reads_trial", "int(row['trial_index']) / 10")
    with pytest.raises(MissingStimulusColumns) as excinfo:
        make_stim_data(model, STIMULUS_ROWS)
    assert excinfo.value.missing == ("trial_index",)


def test_the_same_hook_binds_rows_that_carry_the_column(tmp_path):
    model = _hook_model(tmp_path, "reads_pid", "int(row['participant_id']) % 2")
    rows = [{**row, "participant_id": pid} for pid, row in enumerate(STIMULUS_ROWS)]
    data = make_stim_data(model, rows)
    np.testing.assert_array_equal(data["signal"], [0.0, 1.0])


def test_a_keyerror_from_the_hooks_own_lookup_is_a_bug_not_a_missing_column(tmp_path):
    """'HTTH' is not in the model's private table: that is the hook's bug."""
    model = _hook_model(tmp_path, "bad_table", "_TABLE[row['sequence_a']]")
    with pytest.raises(KeyError) as excinfo:
        make_stim_data(model, STIMULUS_ROWS)
    assert not isinstance(excinfo.value, MissingStimulusColumns)
    assert excinfo.value.args == ("HTTH",)


def test_a_bookkeeping_keyerror_on_rows_that_carry_the_column_propagates(tmp_path):
    """Every row has participant_id, so the KeyError cannot mean it is missing."""
    model = _hook_model(tmp_path, "self_inflicted", "{}['participant_id']")
    rows = [{**row, "participant_id": 0} for row in STIMULUS_ROWS]
    with pytest.raises(KeyError) as excinfo:
        make_stim_data(model, rows)
    assert not isinstance(excinfo.value, MissingStimulusColumns)
