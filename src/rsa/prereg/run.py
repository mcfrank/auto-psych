"""Run the pre-registered experiment: the frozen design, new people, the frozen analysis.

    python -m src.rsa.prereg.run --run-dir <dir> --private-dir <cell>/private \
        --design-dir <the prereg design> --design-sha256 <the pre-registered hash> \
        --predictions data/rsa/live/finals_pool_preds_final.json data/rsa/live/bar_pool_preds.json \
        --participants 1200 --collection live --prolific-mode live --confirm-live-recruitment ...

One experiment, through the campaign's own stages (`src.rsa.outer.run.OuterRun`:
deploy, Prolific, collection, catch exclusions, ``data/modelled.csv``), with
no design step and no inner loop: the design is the pre-registered one (its
hash is checked before anything is deployed) and the analysis is
`src.rsa.prereg.analysis` on the frozen predictions, written to
``experiment1/prereg_analysis.json``. A resubmitted run resumes (a recorded
study is never published twice).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import tyro

from src.rsa.outer.run import OuterConfig, OuterRun, _write_json
from src.rsa.prereg import analysis
from src.rsa.prereg.design import PROMOTED
from src.rsa.prereg.predictions import load_predictions


@dataclass
class PreregConfig:
    run_dir: Path
    design_dir: Path
    """`src.rsa.prereg.design`'s output (design.json, trial_lists.json, prereg_design.json)."""
    design_sha256: str
    """The pre-registered design's hash: the run refuses any other design."""
    predictions: List[Path]
    """The frozen predictions (the finals' and the bar's)."""
    participants: int
    predictions_sha256: List[str] = field(default_factory=list)
    """The pre-registered predictions files' sha256, in ``predictions``' order: the run refuses others."""
    private_dir: Optional[Path] = None
    collection: Literal["simulated", "live"] = "simulated"
    prolific_mode: Literal["test", "live"] = "test"
    confirm_live_recruitment: bool = False
    firebase_project: Optional[str] = None
    run_label: str = "prereg"
    collection_owner: str = "auto-psych"
    repo_root: Optional[Path] = None
    max_wait_sec: float = 48 * 60 * 60
    """Longer than the campaign's 3 h: the study is several times larger."""
    max_catch_errors: int = 0
    ground_truth: Optional[Path] = None
    """simulated: the model file people answer from."""
    existing_data: Optional[Path] = None
    """simulated: the trials the ground truth is fitted to."""
    finals: List[str] = field(default_factory=list)
    seed: int = 0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    """simulated: the ground truth's fit."""


class PreregRun(OuterRun):
    def __init__(self, cfg: PreregConfig):
        self.prereg = cfg
        doc = json.loads((Path(cfg.design_dir) / "prereg_design.json").read_text())
        if doc["design_sha256"] != cfg.design_sha256:
            raise ValueError(f"{cfg.design_dir} holds design {doc['design_sha256']}, not the pre-registered "
                             f"{cfg.design_sha256}")
        if doc["participants"] != cfg.participants:
            raise ValueError(f"the design has {doc['participants']} trial lists; the run recruits {cfg.participants}")
        if cfg.predictions_sha256:
            found = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cfg.predictions]
            if found != list(cfg.predictions_sha256):
                raise ValueError(f"the predictions files {[str(p) for p in cfg.predictions]} are not the "
                                 f"pre-registered ones ({found} != {cfg.predictions_sha256})")
        self.doc = doc
        outer = OuterConfig(
            run_dir=cfg.run_dir, seeds=PROMOTED, existing_data=Path(cfg.existing_data or cfg.design_dir),
            private_dir=cfg.private_dir, collection=cfg.collection, prolific_mode=cfg.prolific_mode,
            confirm_live_recruitment=cfg.confirm_live_recruitment, firebase_project=cfg.firebase_project,
            run_label=cfg.run_label, collection_owner=cfg.collection_owner, repo_root=cfg.repo_root,
            max_wait_sec=cfg.max_wait_sec, ground_truth=cfg.ground_truth, n_experiments=1,
            participants=cfg.participants, trials=doc["trials"], displays=doc["displays"], n_catch=doc["n_catch"],
            max_catch_errors=cfg.max_catch_errors, scope="all", critique=False, seed=cfg.seed,
            num_warmup=cfg.num_warmup, num_samples=cfg.num_samples, num_chains=cfg.num_chains)
        super().__init__(outer, spawner=None)

    def design(self, n: int) -> Path:
        """The pre-registered trial lists, checked against the design's record."""
        out = self.exp(n) / "design"
        lists_path = out / "trial_lists.json"
        if not lists_path.exists():
            src = Path(self.prereg.design_dir)
            if hashlib.sha256((src / "trial_lists.json").read_bytes()).hexdigest() != self.doc["trial_lists_sha256"]:
                raise ValueError(f"{src / 'trial_lists.json'} is not the design's recorded trial lists")
            lists = json.loads((src / "trial_lists.json").read_text())
            if lists["design_sha256"] != self.prereg.design_sha256:
                raise ValueError("the trial lists are of another design")
            out.mkdir(parents=True, exist_ok=True)
            for name in ("design.json", "prereg_design.json"):
                shutil.copyfile(src / name, out / name)
            shutil.copyfile(src / "trial_lists.json", lists_path)
        return lists_path

    def analyse(self) -> Path:
        out = self.exp(1) / "prereg_analysis.json"
        if not out.exists():
            preds = load_predictions(self.prereg.predictions)
            result = analysis.analyse(self.modelled(1), preds, self.prereg.finals or None)
            _write_json(dict(result, design_sha256=self.prereg.design_sha256), out)
        return out

    def run(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = self.private / "prereg_config.json"
        cfg = {k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
               for k, v in asdict(self.prereg).items()}
        operational = ("prolific_mode", "confirm_live_recruitment", "max_wait_sec", "repo_root")
        strip = lambda d: {k: v for k, v in d.items() if k not in operational}  # noqa: E731
        if record.exists() and strip(json.loads(record.read_text())) != strip(cfg):
            raise ValueError(f"{record} records another configuration; a run is resumed with its own")
        _write_json(cfg, record)
        from src.rsa.live.collect import DraftOnly

        print(f"[prereg] design {self.prereg.design_sha256[:12]}", flush=True)
        self.design(1)
        print(f"[prereg] collect ({self.cfg.collection}, {self.cfg.participants} people)", flush=True)
        try:
            self.collect(1)
        except DraftOnly as stop:
            print(f"[prereg] stopped: {stop}", flush=True)
            return
        out = self.analyse()
        result = json.loads(out.read_text())
        print(f"[prereg] best promoted seed: {result['best_promoted']}", flush=True)
        for f, t in result["primary"].items():
            print(f"  {f}: {t['diff']:+.1f} ± {t['se']:.1f} (z {t['z']:.2f}, p {t['p_one_sided']:.4f}) "
                  f"{'rejects' if t['holm_reject'] else 'does not reject'} H0 (Holm)", flush=True)


def main(cfg: PreregConfig) -> None:
    PreregRun(cfg).run()


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(PreregConfig))
