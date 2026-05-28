"""Generate the fixture responses.csv by simulating from bayesian_fair_coin
with known parameters. Reproducible via numpy seed=42.

Run from repo root:
    python3 tests/fixtures/pymc_models/simulate.py
"""
import csv
from pathlib import Path

import numpy as np

THETA = 0.7   # bias of the alternative hypothesis
TAU = 2.0     # softmax temperature
N_TRIALS = 30
SEED = 42


def simulate():
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_TRIALS):
        n_a = int(rng.integers(8, 25))
        n_b = int(rng.integers(8, 25))
        h_a = int(rng.binomial(n_a, rng.uniform(0.3, 0.8)))
        h_b = int(rng.binomial(n_b, rng.uniform(0.3, 0.8)))

        log_fair_a = n_a * np.log(0.5)
        log_bias_a = h_a * np.log(THETA) + (n_a - h_a) * np.log(1 - THETA)
        lbf_a = log_fair_a - log_bias_a

        log_fair_b = n_b * np.log(0.5)
        log_bias_b = h_b * np.log(THETA) + (n_b - h_b) * np.log(1 - THETA)
        lbf_b = log_fair_b - log_bias_b

        p_left = 1.0 / (1.0 + np.exp(-TAU * (lbf_a - lbf_b)))
        chose_left = int(rng.uniform() < p_left)
        rows.append({"n_a": n_a, "h_a": h_a, "n_b": n_b, "h_b": h_b, "chose_left": chose_left})
    return rows


if __name__ == "__main__":
    out_path = Path(__file__).parent / "responses.csv"
    rows = simulate()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_a", "h_a", "n_b", "h_b", "chose_left"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} trials to {out_path}")
