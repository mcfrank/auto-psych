"""The sweep overview builds from a sweep's committed outputs (no fitting)."""

import json

from src.rsa.sweep_report import build, render
from src.runtime.config import REPO_ROOT

SWEEP = REPO_ROOT / "data" / "rsa" / "sherlock_run1"


def test_run1_overview_builds_from_the_committed_cells():
    b = build(SWEEP)
    cells = {c["cell"]: c for c in b["cells"]}
    assert set(cells) == {"real_rep1", "real_rep2", "recovery_literal_rep1", "recovery_literal_rep2",
                          "recovery_salience_rep1", "recovery_salience_rep2"}
    rep1 = cells["real_rep1"]
    assert rep1["exported"] == "surprisal_multimodal_l2_listener_2" and rep1["recovery"] is None
    # Every model scored before the prune has its held-out number, relative to the best seed.
    exported = next(m for m in rep1["models"] if m["status"] == "exported")
    assert exported["insample"] == 0.0 and round(exported["heldout"], 1) == -7.5
    assert max(m["heldout"] for m in rep1["models"]) > 60  # pruned models generalised better
    # Per-source grouped CV (re-scored after run 1): each source's best model.
    assert rep1["cv_origin"] and {sp["source"] for sp in rep1["specialists"]} == {
        "mayn_demberg_2022", "mayn_demberg_2023", "mayn_demberg_2026", "pragmods", "sikos_2021"}
    assert all(m["cv_by_source"] for m in rep1["models"])
    rec = cells["recovery_salience_rep2"]["recovery"]
    assert rec["ground_truth"] == "rsa_l1_salience" and rec["recovered"] and rec["best_seed_rmse"] == 0.049
    html = render(b)
    assert "__BUNDLE_JSON__" not in html and "<title>RSA run 1</title>" in html
    json.loads(html.split('type="application/json">', 1)[1].split("</script>", 1)[0])
