"""The Cloud Functions' list assignment and JSON results (functions/lists.js), run with node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
LISTS = Path(__file__).resolve().parents[1] / "functions" / "lists.js"
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script: str):
    out = subprocess.run([NODE, "-e", f"const L = require({json.dumps(str(LISTS))});\n{script}"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_participants_get_lists_in_arrival_order_and_the_same_list_on_a_reload():
    got = _node("""
      let state = null; const prior = {}; const out = [];
      for (const key of ["a", "b", "c", "a", "d"]) {
        const plan = L.assignmentPlan(state, prior[key] || null, 3);
        if (plan.nextState) { state = plan.nextState; prior[key] = { list_index: plan.listIndex }; }
        out.push(plan.listIndex);
      }
      console.log(JSON.stringify({ out, state }));
    """)
    assert got["out"] == [0, 1, 2, 0, 0]  # "a" again gets 0; "d" wraps round to 0
    assert got["state"] == {"n_lists": 3, "next": 4}


def test_another_list_count_for_the_same_session_is_refused():
    got = _node("""
      try { L.assignmentPlan({ n_lists: 200, next: 5 }, null, 199); console.log("null"); }
      catch (e) { console.log(JSON.stringify(e.message)); }
    """)
    assert "200 lists, not 199" in got


def test_assign_requests_are_validated():
    got = _node("""
      console.log(JSON.stringify([
        L.validateAssign({ collection_session_id: "s", n_lists: 3, participant_key: "p" }),
        L.validateAssign({ n_lists: 3, participant_key: "p" }),
        L.validateAssign({ collection_session_id: "s", n_lists: 0, participant_key: "p" }),
        L.validateAssign({ collection_session_id: "s", n_lists: 3 }),
      ]));
    """)
    assert got[0] == "" and all(got[1:])


def test_json_results_keep_every_trial_whole():
    got = _node("""
      const docs = [{ id: "P1", data: () => ({ prolific_pid: "P1", list_index: 4,
                                              trials: [{ task: "rsa_choice", objects: [[0, 1]] }] }) }];
      console.log(JSON.stringify(L.responsesToJson(docs)));
    """)
    assert got == [{"participant_id_str": "P1", "prolific_pid": "P1", "prolific_study_id": None,
                    "prolific_session_id": None, "submitted_at_client": None, "list_index": 4,
                    "trials": [{"task": "rsa_choice", "objects": [[0, 1]]}]}]
