"""The noise baseline and leaders test: rebuild each theory's out-of-fold scores for each method from the tasks' files, and
apply the loop's own standing rule. Writes aggregates only. Run after the arrays: python opt_analyze.py <opt dir>"""
import json, sys
from pathlib import Path

import numpy as np

from src.rsa_open.corpus import CorpusCV, standing
from src.rsa_open.loop import fitting as ft

BASE = Path(sys.argv[1])
THEORIES = ["bipolar_boundary_spillover_topic_tom_2", "polar_grounding_boundary_spill_tom", "aspect_qud_boundary_tier_shadowing_tom",
            "structural_alignment_pragmatic_tom", "rsa_l2_singleton_feat_color_valence_l0"]
METHODS = {"chain seed 1": "chain_settings_w300_s1", "reuse seed 0": "reuse_w100_s0", "reuse seed 1": "reuse_w100_s1"}
PRUNE = 2.0
blocks = ft.read_corpus(BASE / "input" / "corpus.json")


def scores(theory, method, column):
    out = {b.name: np.full(b.n, np.nan) for b in blocks}
    for k in range(5):
        with np.load(BASE / "pointwise" / f"{theory}__{k}__{method}.npz") as z:
            for b in blocks:
                if f"{b.name}|rows" in z:
                    out[b.name][z[f"{b.name}|rows"]] = z[f"{b.name}|{column}"]
    missing = {n: int(np.isnan(v).sum()) for n, v in out.items() if np.isnan(v).any()}
    if missing:
        raise RuntimeError(f"{theory} {method}: trials without a score {missing}")
    return out


runs = {"chain seed 0": {t: scores(t, METHODS["chain seed 1"], "chain") for t in THEORIES}}
runs.update({label: {t: scores(t, m, "new") for t in THEORIES} for label, m in METHODS.items()})
newest = blocks[-1].name
report = dict(newest_block=newest, trials={b.name: b.n for b in blocks}, runs={})
for label, by_theory in runs.items():
    s = standing({t: CorpusCV(v, True) for t, v in by_theory.items()}, blocks)
    report["runs"][label] = {t: dict(sel_elpd=round(s[t]["sel_elpd"], 2), sel_diff=round(s[t]["sel_diff"], 2), sel_dse=round(s[t]["sel_dse"], 2),
                                     pruned=bool(s[t]["sel_diff"] > PRUNE * s[t]["sel_dse"]),
                                     guard_failures_among_these=s[t]["guard_failures"],
                                     by_block={n: round(d["elpd_cv"], 2) for n, d in s[t]["by_block"].items()}) for t in THEORIES}
base = report["runs"]["chain seed 0"]
report["change_from_chain_seed_0"] = {
    label: {t: dict(newest=round(r[t]["sel_elpd"] - base[t]["sel_elpd"], 2),
                    all_blocks=round(sum(r[t]["by_block"].values()) - sum(base[t]["by_block"].values()), 2)) for t in THEORIES}
    for label, r in report["runs"].items() if label != "chain seed 0"}
(BASE / "results2" / "leaders.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
