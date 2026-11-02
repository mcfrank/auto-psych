"""Print the README's tables from this directory's records (run from this directory).

    python readme_tables.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
GTS = ["tenenbaum_rules_similarity", "bigelow_piantadosi_mixture", "bigelow_piantadosi_independent"]
SEEDS = set(GTS)

summary = json.loads((HERE / "test_retest.json").read_text())

print("| ground truth | repeat | final best model | Pearson r | RMSE | KL regret | best changes (of 18 steps) | first agent-written best |")
print("|---|---|---|---|---|---|---|---|")
for gt in GTS:
    for run in ["run1", "run2", "run3"]:
        steps = json.loads((HERE / run / gt / "trajectory.json").read_text())["trajectory"]
        final = steps[-1]
        best = [s["best_model"] for s in steps]
        changes = sum(a != b for a, b in zip(best, best[1:]))
        first_agent = next(s["global_step"] for s in steps if s["best_model"] not in SEEDS)
        print(f"| `{gt}` | {run[-1]} | `{final['best_model']}` | {final['pearson_r']:.3f} | {final['rmse']:.3f} "
              f"| {final['kl_regret']:.4f} | {changes} | step {first_agent} |")

print()
print("| ground truth | Pearson r, mean (sd) over 3 repeats | min | max |")
print("|---|---|---|---|")
for gt in GTS:
    g = summary["per_gt_model"][gt]
    print(f"| `{gt}` | {g['mean']:.3f} ({g['sd']:.3f}) | {g['min']:.3f} | {g['max']:.3f} |")
print()
print(f"ICC(2,1) of Pearson r across repeats: {summary['icc_2_1']:.2f}; "
      f"mean pairwise correlation between repeats: {summary['mean_pairwise_corr']:.2f}")

print()
print("| ground truth | experiment | loop's best, Pearson r | fitted seeds, Pearson r | loop's best, RMSE | fitted seeds, RMSE |")
print("|---|---|---|---|---|---|")
rows = {(b["gt_model"], b["experiment"], b["metric"]): b for b in summary["loop_vs_fitted_baseline"]}
for gt in GTS:
    for exp in (1, 2, 3):
        r, e = rows[(gt, exp, "pearson_r")], rows[(gt, exp, "rmse")]
        print(f"| `{gt}` | {exp} | {r['loop_mean']:.3f} | {r['fitted_baseline_mean']:.3f} "
              f"| {e['loop_mean']:.3f} | {e['fitted_baseline_mean']:.3f} |")
