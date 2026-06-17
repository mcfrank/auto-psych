import json
import itertools
import random

random.seed(42)

def generate_sequences():
    sequences = []
    for length in range(3, 9):
        seqs = ["".join(seq) for seq in itertools.product("HT", repeat=length)]
        sequences.extend(seqs)
    return sequences

all_seqs = generate_sequences()

candidates = []
# Ensure a good mix of lengths and head proportions
# Let's do random pairs
for _ in range(300):
    a = random.choice(all_seqs)
    b = random.choice(all_seqs)
    if a != b:
        candidates.append({"sequence_a": a, "sequence_b": b})

with open("/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_heads_more_random/experiment3/design/candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)

print("Generated 300 candidate pairs.")
