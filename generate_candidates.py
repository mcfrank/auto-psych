import json
import random

def random_seq(length):
    return "".join(random.choices(["H", "T"], k=length))

candidates = []
for _ in range(300):
    len_a = random.randint(4, 8)
    len_b = random.randint(4, 8)
    a = random_seq(len_a)
    b = random_seq(len_b)
    while a == b:
        b = random_seq(len_b)
    candidates.append({"sequence_a": a, "sequence_b": b})

out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/longer_runs_more_random/experiment1/design/candidates.json"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(candidates, f, indent=2)
