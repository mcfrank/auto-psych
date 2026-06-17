import json
import random
import itertools

# generate all sequences of length 4 to 8
lengths = range(4, 9)
all_seqs = []
for L in lengths:
    for seq in itertools.product('HT', repeat=L):
        all_seqs.append("".join(seq))

print(f"Total sequences: {len(all_seqs)}")

candidates = []
# generate random pairs
random.seed(42)
seen = set()

while len(candidates) < 5000:
    seq_a = random.choice(all_seqs)
    seq_b = random.choice(all_seqs)
    
    if seq_a == seq_b:
        continue
        
    # ensure no identical pairs in reverse
    pair_id = tuple(sorted([seq_a, seq_b]))
    if pair_id in seen:
        continue
        
    seen.add(pair_id)
    candidates.append({"sequence_a": seq_a, "sequence_b": seq_b})

out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/encoding_compressibility/experiment3/design/candidates.json"
with open(out_path, "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Wrote {len(candidates)} candidates to {out_path}")
