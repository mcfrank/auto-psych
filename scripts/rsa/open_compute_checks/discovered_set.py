"""The seeds and discovered theories scored beside the standard-RSA grid (docs/auto_open/RSA_GRID.md, PI 2026-11-16).

    uv run python scripts/rsa/open_compute_checks/discovered_set.py            # writes data/rsa_open/grid/discovered/
    uv run python scripts/rsa/open_compute_checks/discovered_set.py check [DIR NAME...]   # gates and contracts, before any fit

Every theory as an open-format file (`PARAMS`, `listener`), with a `models_manifest.yaml` giving each one's group,
origin and **lineage**: the data the theorists who made it, or the loop that selected it, had seen. The page marks each
(theory, study) score as seen or unseen from that.

- **seeds**: the closed campaign's bar (5 starting models and the promoted seeds, already ported in
  `data/rsa_open/seeds/bar`), less the three that are grid corners exactly (`literal_listener`, `rsa_l1`, `rsa_l2`:
  their grid fits stand for them). Seen: the closed campaign's literature block, which left out pragmods' valence,
  familiarization, colour and greyscale displays (`docs/auto_rsa/LOOP_ALGORITHM.md`, E).
- **closed**: each closed chain's exported best theory after each live experiment (3 chains x 3), ported by
  `src.rsa_open.port` (mechanical; it refuses rather than guesses) and otherwise unchanged: the closed campaign
  had no rule against hand-set numbers, so they stay, and the manifest counts them (`hand_set_numbers`). Seen: the
  plain literature and that chain's experiments so far.
- **consolidation**: the closed consolidation's full superset and its backward-elimination survivor
  (`docs/auto_rsa/CONSOLIDATION.md`), each wrapped (`WRAPPER`: the closed model at lapse 0, the link's lapse
  in its place, as the port's rule (a); the port cannot read their `make()`-built PARAMS). Seen: the plain literature and all nine closed experiments
  (not the pre-registered final). Both files carry the superset's whole source with mechanisms switched on by a
  footer, so code-size complexity overstates them (flagged).
- **open pilot 1** and **stage 1** (open and fixed arms): each experiment's incumbent, already in the open format.
  Seen: the whole literature block (the open campaign's E keeps every display) and the chain's experiments so far.

Study labels follow `rsa_grid.py`'s corpus: literature sources by name (pragmods split at report time into
`pragmods_trials` and `pragmods_prior`), `closed_c{c}e{e}`, `closed_final`, `pilot1_exp{n}`, `s1_open_exp{n}`,
`s1_fixed_exp{n}`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "data/rsa_open/grid/discovered"
GRID_CORNERS = {"literal_listener", "rsa_l1", "rsa_l2"}
CLOSED = REPO / "data/rsa/live"
OPEN_PILOT = REPO / "data/rsa_open/live/open_chain_0"
STAGE1 = REPO / "data/rsa_open/live_v2"
LEAN = REPO / "data/rsa/consolidate/eliminate/lean"


def _best(loop_dir: Path) -> str:
    history = json.loads((loop_dir / "history.json").read_text())
    return (history[-1] if isinstance(history, list) else history["steps"][-1])["best_model"]


def _rationale(models_dir: Path, name: str) -> str:
    manifest = models_dir / "models_manifest.yaml"
    if manifest.exists():
        for e in yaml.safe_load(manifest.read_text())["models"]:
            if e["name"] == name:
                return " ".join(str(e.get("rationale") or e.get("hypothesis") or "").split())
    return ""


def _ported(path: Path, where: str) -> str:
    """The port, unchanged otherwise: the closed campaign had no rule against hand-set numbers, so its theories keep
    theirs (the manifest counts them, `hand_set_numbers`); they are scored as they were discovered."""
    from src.rsa_open.port import port_file

    return port_file(path)


WRAPPER = """

# --- open-format wrapper (scripts/rsa/open_compute_checks/discovered_set.py) ---
# The consolidation builds PARAMS and choice_probs with make(), which the mechanical port cannot read. The theory is
# the closed model at lapse 0 (with_lapse(p, 0) is p); the response link supplies the lapse, as the port's rule (a).
_CLOSED_PARAMS = PARAMS
PARAMS = {k: v for k, v in _CLOSED_PARAMS.items() if k != "lapse"}
_closed_choice_probs = choice_probs
del choice_probs


def listener(params, ctx):
    return _closed_choice_probs({**params, "lapse": 0.0}, ctx)
"""


def _consolidated(source: str) -> str:
    if "PARAMS, choice_probs = make(MECHANISMS_ON)" not in source:
        raise ValueError("not a consolidation variant file: no 'PARAMS, choice_probs = make(MECHANISMS_ON)'")
    return source + WRAPPER


def entries() -> List[Dict]:
    out: List[Dict] = []
    bar = REPO / "data/rsa_open/seeds/bar"
    for e in yaml.safe_load((bar / "models_manifest.yaml").read_text())["models"]:
        if e["name"] in GRID_CORNERS:
            continue
        out.append(dict(name=e["name"], group="seed", origin=f"data/rsa_open/seeds/bar/{e['name']}.py",
                        source=(bar / f"{e['name']}.py").read_text(), rationale=" ".join(str(e.get("rationale", "")).split()),
                        seen={"literature": "plain", "studies": []}))
    for c in range(3):
        for x in range(1, 4):
            loop = CLOSED / f"live_rsa_c{c}/outputs/experiment{x}/model_loop"
            name = _best(loop)
            path = loop / "models" / f"{name}.py"
            out.append(dict(name=f"closed_c{c}e{x}_{name}", group="closed", origin=str(path.relative_to(REPO)),
                            source=_ported(path, str(path)), rationale=_rationale(loop / "models", name),
                            seen={"literature": "plain", "studies": [f"closed_c{c}e{k}" for k in range(1, x + 1)]}))
    from src.rsa.consolidate.superset import MECHANISMS
    from src.rsa.consolidate.variants import write_variant

    with tempfile.TemporaryDirectory() as tmp:
        name, path = write_variant(MECHANISMS, Path(tmp))
        out.append(dict(name=name, group="consolidation", origin=f"src.rsa.consolidate.variants.write_variant(all {len(MECHANISMS)} mechanisms)",
                        source=_consolidated(path.read_text()), rationale="The closed consolidation's superset: every mechanism switched on.",
                        seen={"literature": "plain", "studies": [f"closed_c{c}e{x}" for c in range(3) for x in range(1, 4)]},
                        whole_superset=True))
    for path in sorted(LEAN.glob("*.py")):
        out.append(dict(name=path.stem, group="consolidation", origin=str(path.relative_to(REPO)),
                        source=_consolidated(path.read_text()), rationale=_rationale(LEAN, path.stem) or "Backward elimination's survivor.",
                        seen={"literature": "plain", "studies": [f"closed_c{c}e{x}" for c in range(3) for x in range(1, 4)]},
                        whole_superset=True))
    for x in (1, 2):
        loop = OPEN_PILOT / f"experiment{x}/loop"
        name = _best(loop)
        path = loop / "models" / f"{name}.py"
        out.append(dict(name=f"pilot1_e{x}_{name}", group="open pilot 1", origin=str(path.relative_to(REPO)),
                        source=path.read_text(), rationale=_rationale(loop / "models", name),
                        seen={"literature": "all", "studies": [f"pilot1_exp{k}" for k in range(1, x + 1)]}))
    for arm, prefix in (("open", "s1_open"), ("fixed", "s1_fixed")):
        for x in (1, 2, 3):
            loop = STAGE1 / f"{arm}_chain_0/experiment{x}/loop"
            name = _best(loop)
            path = loop / "models" / f"{name}.py"
            out.append(dict(name=f"{arm}{x}_{name}", group=f"stage 1 {arm}", origin=str(path.relative_to(REPO)),
                            source=path.read_text(), rationale=_rationale(loop / "models", name),
                            seen={"literature": "all", "studies": [f"{prefix}_exp{k}" for k in range(1, x + 1)]}))
    return out


def write(out_dir: Path = OUT) -> Path:
    from src.rsa_open.complexity import hand_set_numbers

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, seen_hash = [], {}
    for e in entries():
        digest = hashlib.sha256(e["source"].encode()).hexdigest()
        if digest in seen_hash:
            raise ValueError(f"{e['name']} is the same file as {seen_hash[digest]}: keep one")
        seen_hash[digest] = e["name"]
        (out_dir / f"{e['name']}.py").write_text(e["source"])
        rows.append({k: v for k, v in e.items() if k != "source"}
                    | {"hand_set_numbers": hand_set_numbers(e["source"]), "sha256": digest})
    (out_dir / "models_manifest.yaml").write_text(yaml.safe_dump({"models": rows}, sort_keys=False, width=120))
    print(f"[discovered] {len(rows)} theories -> {out_dir}")
    for g in dict.fromkeys(r["group"] for r in rows):
        print(f"  {g}: {sum(r['group'] == g for r in rows)}")
    return out_dir




def check(theory_dir: Path = OUT, only=(), n_pool: int = 60, n_lit: int = 80) -> None:
    """Before any fit: every theory passes the code and shape gates, loads, and meets the listener and speaker
    contracts (valid distributions, finite gradients) on displays from the design pool and the committed literature
    (pragmods' valence, familiarization and colour displays included). Raises on the first failure, naming it."""
    import numpy as np

    from src.rsa.context import group_by_shape
    from src.rsa.dataset import load_forced_choice
    from src.rsa.loop.code_gate import code_problems
    from src.rsa.loop.novelty import plain_pool
    from src.rsa.model_file import check_contract
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop.gates import check_speaker_contract, theory_shape_problems

    pool = plain_pool()
    contexts = [pool[i] for i in np.linspace(0, len(pool) - 1, n_pool).round().astype(int)]
    data = REPO / "src/pipelines/outer_loop/projects/rsa_reference/data"
    for csv in sorted(data.glob("*_trials.csv")):
        t = load_forced_choice(csv)
        special = [c for c in t.contexts if c.familiarization is not None or c.grayscale is not None or c.valence]
        plain = [c for c in t.contexts if c not in special]
        rng = np.random.default_rng(0)
        for group in (special, plain):
            if group:
                contexts += [group[i] for i in rng.choice(len(group), min(len(group), n_lit // 2), replace=False)]
    groups = group_by_shape(contexts)
    names = [e["name"] for e in yaml.safe_load((theory_dir / "models_manifest.yaml").read_text())["models"]]
    if only:  # a subset by name (one process per theory keeps memo's compiled code from piling up)
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(f"not in the manifest: {unknown}")
        names = [n for n in names if n in only]
    for name in names:
        source = (theory_dir / f"{name}.py").read_text()
        problems = code_problems(source) + theory_shape_problems(source)
        if problems:
            raise ValueError(f"{name}: {problems}")
        theory = TheoryModel(theory_dir / f"{name}.py", name=name)
        check_contract(theory, groups)
        check_speaker_contract(theory, groups)
        print(f"[check] {name}: ok ({len(theory.params)} parameters)", flush=True)
    print(f"[check] all {len(names)} theories pass on {len(contexts)} displays")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        check(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT, only=tuple(sys.argv[3:]))
    else:
        write(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
