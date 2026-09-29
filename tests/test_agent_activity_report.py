"""What the agents looked at, for a person to skim after a run.

Agents may look up published papers (fine, and interesting to see), and the
sandbox keeps them out of other runs, but nothing used to record either. After
a run, every URL in the agents' logs and every absolute path they touched
outside their own directory (and the system software) is listed per cell.
Warn-only: nothing here fails a cell.
"""

from __future__ import annotations

import json

from scripts.subjective_randomness.slurm.agent_activity_report import activity


def test_urls_and_outside_paths_are_collected_with_counts(tmp_path):
    agent_dir = tmp_path / "agent_trees" / "abc"
    logs = agent_dir / "repo" / "_runs" / "cell_1" / "iter_0" / "candidate_0"
    logs.mkdir(parents=True)
    lines = [
        {"tool": "webfetch", "input": {"url": "https://psycnet.apa.org/record/1997-03707-001"}},
        {"tool": "bash", "input": {"command": f"cat {agent_dir}/repo/_runs/cell_1/models/m.py"}},
        {"tool": "bash", "input": {"command": "ls /scratch/users/someone/auto-psych/sweep/run1"}},
        {"tool": "bash", "input": {"command": "cat /tmp/explore.py; ls /share/software/modules"}},
        {"tool": "webfetch", "input": {"url": "https://psycnet.apa.org/record/1997-03707-001"}},
        {"tool": "bash", "input": {"command": "cat /home/users/someone/.claude/tool-results/x"}},
    ]
    (logs / "agent.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    urls, outside = activity(agent_dir, also_expected=["/home/users/someone"])
    assert urls == {"https://psycnet.apa.org/record/1997-03707-001": 2}
    assert outside == {"/scratch/users/someone/auto-psych/sweep/run1": 1}
