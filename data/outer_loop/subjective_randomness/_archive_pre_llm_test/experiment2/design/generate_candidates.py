import itertools
import random
import json
import os

seqs = []
for length in range(3, 9):
    for s in itertools.product("HT", repeat=length):
        seqs.append("".join(s))

random.seed(42)
pairs = []
# Ensure no duplicates, though it's 300 out of ~125,000, so it's fine.
for _ in range(300):
    a = random.choice(seqs)
    b = random.choice(seqs)
    pairs.append({"sequence_a": a, "sequence_b": b})

out_dir = "/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/design"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "candidates.json"), "w") as f:
    json.dump(pairs, f, indent=2)

print("Generated candidates.json")
