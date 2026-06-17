import json
import random

def generate_sequence(length):
    return "".join(random.choice(['H', 'T']) for _ in range(length))

def generate_candidates(num_candidates=300, min_len=3, max_len=8):
    candidates = []
    seen = set()
    
    while len(candidates) < num_candidates:
        len_a = random.randint(min_len, max_len)
        len_b = random.randint(min_len, max_len)
        
        seq_a = generate_sequence(len_a)
        seq_b = generate_sequence(len_b)
        
        # Don't use identical sequences
        if seq_a == seq_b:
            continue
            
        # Avoid exact duplicates
        pair = (seq_a, seq_b)
        pair_rev = (seq_b, seq_a)
        if pair in seen or pair_rev in seen:
            continue
            
        seen.add(pair)
        candidates.append({"sequence_a": seq_a, "sequence_b": seq_b})
        
    return candidates

# Also try to include some specific variations like different lengths, identical lengths, extremes
candidates = generate_candidates(300, 3, 8)

out_file = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment1/design/candidates.json"

with open(out_file, "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Wrote {len(candidates)} candidates to {out_file}")
