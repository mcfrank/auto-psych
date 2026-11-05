"""Print the README's tables from this directory's records (run from anywhere).

    python readme_tables.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = {"tenenbaum_rules_similarity", "bigelow_piantadosi_mixture", "bigelow_piantadosi_independent"}
RUNS = ["run1", "run2", "run3"]

for run in RUNS:
    steps = json.loads((HERE / "harness" / run / "trajectory.json").read_text())["trajectory"]
    best = [s["best_model"] for s in steps]
    changes = sum(a != b for a, b in zip(best, best[1:]))
    first_agent = next(s["global_step"] for s in steps if s["best_model"] not in SEEDS)
    print(f"**{run}**: the best model changed {changes} times in {len(steps)} scoring steps; "
          f"the first agent-written best was at step {first_agent}.")
    print()
    print("| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) "
          "| lead over runner-up (dse) | carried |")
    print("|---|---|---|---|---|---|---|")
    for exp in (1, 2, 3):
        post = json.loads((HERE / "collected" / run / "number_game" / f"experiment{exp}" / "model_loop"
                           / "model_posterior.json").read_text())
        start = next(s["best_model"] for s in steps if s["experiment"] == exp and s["step"] == 0)
        end = post["best_model"]
        comp = post["comparison"]
        runner = [c for c in comp.values() if c["rank"] == 1]
        lead = f"{runner[0]['elpd_diff']:.1f} ({runner[0]['dse']:.1f})" if runner else "n/a"
        elpd = comp[end]["elpd_loo"]
        print(f"| {exp} | {post['n_trials']:,} | `{start}` | `{end}` | {elpd:.1f} ({elpd / post['n_trials']:.3f}) "
              f"| {lead} | {len(comp)} |")
    print()

unseen = HERE / "unseen"
if unseen.exists():
    for run in RUNS:
        meta = json.loads((unseen / run / "unseen_items.json").read_text())
        print(f"**{run}**: {meta['n_unseen_items']:,} unseen items, {meta['n_unseen_responses']:,} responses "
              f"(yes rate {meta['unseen_yes_rate']:.3f}); noise ceiling R² {meta['r2_ceiling']:.2f}.")
        print()
        print("| model | log-lik per response | item R² | % of ceiling |")
        print("|---|---|---|---|")
        for r in csv.DictReader((unseen / run / "unseen_scores.csv").open()):
            r2 = f"{float(r['item_r2']):.2f}" if r["item_r2"] else "—"
            share = f"{float(r['share_of_ceiling']):.0%}" if r["share_of_ceiling"] else "—"
            name = r["model"] if r["model"].startswith("constant") else f"`{r['model']}`"
            print(f"| {name} | {float(r['loglik_per_response']):.3f} | {r2} | {share} |")
        print()
