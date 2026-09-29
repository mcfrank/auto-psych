"""Which holdout cells to resume, and which are missing from a sweep.

Cells used to die on a time limit or out of memory and then silently vanish
from every report. After each array, a retry job resumes the failed tasks
(out-of-memory ones with more memory), and the summary job lists every
expected cell that has no result.
"""

from __future__ import annotations

import json

from scripts.subjective_randomness.slurm.cell_status import missing_cells, retry_plan

SACCT = """\
45287317_1|COMPLETED
45287317_2|TIMEOUT
45287317_3|OUT_OF_MEMORY
45287317_4|FAILED
45287317_5|CANCELLED by 389162
45287317_6|NODE_FAIL
"""


def test_failed_tasks_are_resumed_and_out_of_memory_ones_get_more_memory():
    plan = retry_plan(SACCT)
    assert plan == {"same_memory": [2, 4, 6], "more_memory": [3]}


def test_a_task_the_user_cancelled_is_not_resumed():
    assert 5 not in retry_plan(SACCT)["same_memory"] + retry_plan(SACCT)["more_memory"]


def test_a_clean_array_needs_no_retry():
    assert retry_plan("1_1|COMPLETED\n1_2|COMPLETED\n") == {"same_memory": [], "more_memory": []}


def test_missing_cells_are_every_expected_cell_without_a_result(tmp_path):
    (tmp_path / "run1" / "motif_stack").mkdir(parents=True)
    (tmp_path / "run1" / "motif_stack" / "holdout.json").write_text(json.dumps({}))
    (tmp_path / "run2" / "motif_stack").mkdir(parents=True)  # started, no result
    assert missing_cells(tmp_path, n_repeats=2, gt_models=["falk_konold_dp", "motif_stack"]) == [
        "run1/falk_konold_dp",
        "run2/falk_konold_dp",
        "run2/motif_stack",
    ]
