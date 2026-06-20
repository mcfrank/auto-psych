"""Load a unified pilot config (pilot.yaml) for run_pilot.sh.

Responsibilities:
  * validate the config;
  * render the `prolific:` block to projects/<project>/prolific_config.yaml
    (the file the pipeline reads), unless --check is passed;
  * print a cost/scope summary and validate the Prolific token (to STDERR);
  * emit run + modeling settings as `export KEY=value` lines (to STDOUT) for the
    launcher to eval.

Exits non-zero on any problem so the launcher aborts before money is spent.

Usage: python _pilot_config.py <pilot.yaml> [--check]
"""

import shlex
import sys
from pathlib import Path

import yaml

# Run standalone (python <file>): put the repo root on sys.path so `src` imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.prolific import get_me, load_prolific_config  # noqa: E402
from src.pipelines.outer_loop.deployment.prolific import compute_reward_cents  # noqa: E402

PROLIFIC_SERVICE_FEE = 0.33  # ~33%; verify the current rate in your Prolific account
VALID_DESIGN_MODES = {"exhaustive", "agent"}
VALID_CODING_AGENTS = {"opencode", "claude"}


def die(msg: str) -> "None":
    sys.exit(f"pilot config error: {msg}")


def req(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, ""):
        die(f"missing required `{where}{key}`")
    return d[key]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    check_only = "--check" in sys.argv[1:]      # validate + cost + env, do NOT render
    render_only = "--render-only" in sys.argv[1:]  # render prolific_config into THIS repo, nothing else
    if not args:
        die("usage: _pilot_config.py <pilot.yaml> [--check | --render-only]")
    cfg_path = Path(args[0])
    if not cfg_path.is_file():
        die(f"config not found: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        die(f"invalid YAML in {cfg_path}: {e}")
    if not isinstance(cfg, dict):
        die(f"{cfg_path} must be a mapping")

    project = str(req(cfg, "project", ""))
    run_label = str(req(cfg, "run_label", ""))
    experiments = int(cfg.get("experiments") or 1)
    if experiments < 1:
        die("`experiments` must be >= 1")
    design_mode = str(cfg.get("design_mode") or "exhaustive")
    if design_mode not in VALID_DESIGN_MODES:
        die(f"`design_mode` must be one of {sorted(VALID_DESIGN_MODES)}")
    coding_agent = str(cfg.get("coding_agent") or "opencode")
    if coding_agent not in VALID_CODING_AGENTS:
        die(f"`coding_agent` must be one of {sorted(VALID_CODING_AGENTS)}")
    firebase_project = str(cfg.get("firebase_project") or "")
    walltime = str(cfg.get("walltime") or "1-00:00:00")
    qos = str(cfg.get("qos") or "")

    pro = cfg.get("prolific") or {}
    if not isinstance(pro, dict):
        die("`prolific` must be a mapping")
    participants = int(req(pro, "participants", "prolific."))
    if participants < 1:
        die("`prolific.participants` must be >= 1")

    mdl = cfg.get("modeling") or {}
    if not isinstance(mdl, dict):
        die("`modeling` must be a mapping")

    # --- render projects/<project>/prolific_config.yaml from the prolific block
    rendered = {k: v for k, v in pro.items() if k != "participants"}
    rendered["total_available_places"] = participants
    pcfg_path = REPO_ROOT / "projects" / project / "prolific_config.yaml"
    if not check_only:
        if not pcfg_path.parent.is_dir():
            die(f"no project dir for {project!r} at {pcfg_path.parent}")
        header = (
            "# AUTO-GENERATED from scripts/outer_loop_live/pilot.yaml by run_pilot.sh.\n"
            "# Do not edit by hand — edit pilot.yaml and re-launch.\n\n"
        )
        pcfg_path.write_text(
            header + yaml.safe_dump(rendered, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    if render_only:
        print(f"[render] wrote {pcfg_path}", file=sys.stderr)
        return

    # --- cost summary (use the loader so it matches what the pipeline will see)
    eff = load_prolific_config(project) if not check_only else {
        **load_prolific_config(project), **rendered
    }
    reward = compute_reward_cents(eff)  # cents/participant
    minutes = float(eff.get("estimated_completion_time") or 5)
    per_study = reward * participants
    per_fee = round(per_study * PROLIFIC_SERVICE_FEE)
    per_total = per_study + per_fee
    grand = per_total * experiments

    w = sys.stderr
    print("=" * 66, file=w)
    print("  PILOT STUDY  —  recruits REAL humans and spends REAL money", file=w)
    print("=" * 66, file=w)
    print(f"  config            : {cfg_path}", file=w)
    print(f"  project           : {project}", file=w)
    print(f"  run label         : {run_label}", file=w)
    print(f"  study name        : {eff.get('name')}", file=w)
    print(f"  task length       : {minutes:g} min", file=w)
    print(f"  reward            : ${reward/100:,.2f}/participant  (~${reward/minutes*60/100:,.2f}/hr)", file=w)
    print(f"  participants (N)  : {participants} per experiment", file=w)
    print(f"  experiments       : {experiments}  (design={design_mode}; coding agent={coding_agent})", file=w)
    print(f"  cost / experiment : ${per_study/100:,.2f} reward + ~${per_fee/100:,.2f} fee = ~${per_total/100:,.2f}", file=w)
    label = "EST. GRAND TOTAL " if experiments > 1 else "ESTIMATED TOTAL  "
    print(f"  {label} : ~${grand/100:,.2f}" + (f"  ({experiments} x {participants})" if experiments > 1 else ""), file=w)
    print(f"  (Prolific fee est ~{int(PROLIFIC_SERVICE_FEE*100)}%; confirm the current rate in your account.)", file=w)

    me, err = get_me()
    if err or not me:
        die(f"Prolific token check FAILED: {err}")
    print(f"  prolific token    : OK (researcher {me.get('id')})", file=w)
    if not check_only:
        print(f"  rendered          : {pcfg_path}", file=w)
    print("=" * 66, file=w)

    # --- machine-readable env for the launcher (STDOUT) ----------------------
    out = {
        "PROJECT": project,
        "RUN_LABEL": run_label,
        "N_EXPERIMENTS": experiments,
        "DESIGN_MODE": design_mode,
        "CODING_AGENT": coding_agent,
        "FIREBASE_PROJECT": firebase_project,
        "WALLTIME": walltime,
        "QOS": qos,
        "N_PARTICIPANTS": participants,
        "INNER_LOOP_ITERATIONS": mdl.get("inner_loop_iterations", ""),
        "INNER_LOOP_CANDIDATES": mdl.get("inner_loop_candidates", ""),
        "DRAWS": mdl.get("draws", ""),
        "TUNE": mdl.get("tune", ""),
        "CHAINS": mdl.get("chains", ""),
    }
    for k, v in out.items():
        print(f"export {k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()
