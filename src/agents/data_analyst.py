"""Data analyst: aggregate participant data (from CSV or JATOS API) and produce summary stats."""

from pathlib import Path
from typing import Any, Dict
import json
import subprocess
import sys

from src.config import agent_dir, REPO_ROOT
from src.console_log import agent_header, log_status


def run_data_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If simulated_data_path is set: read CSV, aggregate in Python, write aggregate.csv and summary_stats.json.
    Else: write R script that uses JATOS API (httr) to fetch results and aggregate; run it if R is available.
    Also write the R script for reproducibility.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    agent_header("6data_analyst", run_id, state.get("total_runs"), state.get("mode"))
    if state.get("validation_retry_count", 0) > 0:
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/3)")
    out_dir = agent_dir(project_id, run_id, "6data_analyst")
    out_dir.mkdir(parents=True, exist_ok=True)

    simulated_path = state.get("simulated_data_path")
    if simulated_path and Path(simulated_path).exists():
        # Use Python to aggregate from local CSV
        aggregate, summary = _aggregate_csv(Path(simulated_path))
        (out_dir / "aggregate.csv").write_text(aggregate, encoding="utf-8")
        (out_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        # JATOS path: write R script and try to run it
        jatos_component_id = state.get("jatos_component_id") or ""
        _write_r_script(out_dir, jatos_component_id)
        r_script = out_dir / "analysis_script.R"
        try:
            subprocess.run(
                [str(_r_binary()), str(r_script)],
                cwd=str(out_dir),
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # No R or script failed: produce minimal aggregate from placeholder
            (out_dir / "aggregate.csv").write_text("stimulus_id,chose_left_pct,n\n0,0.5,0\n", encoding="utf-8")
            (out_dir / "summary_stats.json").write_text(json.dumps({"note": "R not run or JATOS data not available"}), encoding="utf-8")

    # Write R script for reproducibility (handles both simulated CSV and JATOS)
    _write_r_script(out_dir, state.get("jatos_component_id"), simulated_path)

    return {
        **state,
        "summary_stats_path": str(out_dir / "summary_stats.json"),
        "aggregate_csv_path": str(out_dir / "aggregate.csv"),
    }


def _aggregate_csv(csv_path: Path) -> tuple[str, dict]:
    """Aggregate by (sequence_a, sequence_b): proportion chose_left, n."""
    import csv as csv_module
    from collections import defaultdict
    rows = list(csv_module.DictReader(open(csv_path, encoding="utf-8")))
    key_to_left = defaultdict(list)
    for r in rows:
        key = (r["sequence_a"], r["sequence_b"])
        key_to_left[key].append(int(r.get("chose_left", 0)))
    lines = ["sequence_a,sequence_b,chose_left_pct,n\n"]
    for (sa, sb), lefts in sorted(key_to_left.items()):
        pct = sum(lefts) / len(lefts) if lefts else 0
        lines.append(f"{sa},{sb},{pct:.4f},{len(lefts)}\n")
    aggregate = "".join(lines)
    summary = {
        "n_stimuli": len(key_to_left),
        "n_responses_total": len(rows),
        "mean_chose_left": sum(int(r.get("chose_left", 0)) for r in rows) / len(rows) if rows else 0,
    }
    return aggregate, summary


def _write_r_script(out_dir: Path, jatos_component_id: str, simulated_csv_path: str | None = None) -> None:
    """Write R script that fetches from JATOS (if component_id) or reads local CSV and aggregates."""
    script = '''
# Data analyst script: fetch data (JATOS or local CSV) and aggregate
# Usage: Rscript analysis_script.R

library(httr)

# Read secrets (JATOS token)
secrets_file <- file.path("{{REPO_ROOT}}", ".secrets")
token <- NULL
if (file.exists(secrets_file)) {
  lines <- readLines(secrets_file)
  for (line in lines) {
    if (grepl("^JATOS_API_TOKEN=", line)) {
      token <- sub("^JATOS_API_TOKEN=\\s*", "", line)
      break
    }
  }
}

# Option 1: Local CSV (simulated participants)
simulated_csv <- "{{SIMULATED_CSV}}"
if (nchar(simulated_csv) > 0 && file.exists(simulated_csv)) {
  d <- read.csv(simulated_csv)
  agg <- aggregate(chose_left ~ sequence_a + sequence_b, data = d, FUN = function(x) c(mean = mean(x), n = length(x)))
  agg$chose_left_pct <- agg$chose_left[, "mean"]
  agg$n <- agg$chose_left[, "n"]
  agg$chose_left <- NULL
  write.csv(agg, "aggregate.csv", row.names = FALSE)
  summary_stats <- list(
    n_stimuli = nrow(agg),
    n_responses_total = nrow(d),
    mean_chose_left = mean(d$chose_left)
  )
  writeLines(jsonlite::toJSON(summary_stats, auto_unbox = TRUE), "summary_stats.json")
  quit(save = "no", status = 0)
}

# Option 2: JATOS API
component_id <- "{{COMPONENT_ID}}"
if (is.null(token) || nchar(component_id) == 0) {
  writeLines('{"note": "No JATOS token or component_id"}', "summary_stats.json")
  quit(save = "no", status = 0)
}

url <- paste0("https://your-jatos-server/jatos/api/v1/results?componentId=", component_id)
r <- POST(url, add_headers(Authorization = paste("Bearer", token)), write_disk("results.zip", overwrite = TRUE))
if (status_code(r) != 200) {
  writeLines('{"note": "JATOS request failed"}', "summary_stats.json")
  quit(save = "no", status = 1)
}
# Unzip and process... (simplified; real script would unzip and parse)
writeLines('{"note": "JATOS export done; parse results.zip manually or extend script"}', "summary_stats.json")
'''
    script = script.replace("{{REPO_ROOT}}", str(REPO_ROOT).replace("\\", "/"))
    script = script.replace("{{COMPONENT_ID}}", jatos_component_id or "")
    script = script.replace("{{SIMULATED_CSV}}", simulated_csv_path or "")
    (out_dir / "analysis_script.R").write_text(script, encoding="utf-8")


def _r_binary() -> Path:
    """Path to Rscript."""
    import shutil
    return Path(shutil.which("Rscript") or "Rscript")
