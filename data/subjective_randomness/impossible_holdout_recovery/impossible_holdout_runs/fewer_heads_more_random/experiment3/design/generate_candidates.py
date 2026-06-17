import json
import random

def generate_random_sequence(length):
    return "".join(random.choices(["H", "T"], k=length))

candidates = []
seen = set()

# Seed for reproducibility
random.seed(42)

while len(candidates) < 300:
    len_a = random.randint(3, 8)
    len_b = random.randint(3, 8)
    
    seq_a = generate_random_sequence(len_a)
    seq_b = generate_random_sequence(len_b)
    
    if seq_a == seq_b:
        continue
        
    pair_tuple = (seq_a, seq_b)
    if pair_tuple in seen:
        continue
        
    seen.add(pair_tuple)
    candidates.append({"sequence_a": seq_a, "sequence_b": seq_b})

out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/fewer_heads_more_random/experiment3/design/candidates.json"
with open(out_path, "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Wrote {len(candidates)} candidates to {out_path}")
