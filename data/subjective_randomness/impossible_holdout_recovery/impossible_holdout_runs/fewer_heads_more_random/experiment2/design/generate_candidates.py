import json
import random
import itertools

random.seed(42)

def generate_sequences(max_length=8):
    seqs = []
    for length in range(1, max_length + 1):
        for combo in itertools.product(["H", "T"], repeat=length):
            seqs.append("".join(combo))
    return seqs

all_seqs = generate_sequences(8)

candidates = []
# Generate 250 random pairs
for _ in range(300):
    a = random.choice(all_seqs)
    b = random.choice(all_seqs)
    while a == b:
        b = random.choice(all_seqs)
    candidates.append({"sequence_a": a, "sequence_b": b})

with open("/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/fewer_heads_more_random/experiment2/design/candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Generated {len(candidates)} candidates.")
