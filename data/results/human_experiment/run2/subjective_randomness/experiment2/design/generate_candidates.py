import json
import random
import itertools

def generate_all_sequences(min_len, max_len):
    seqs = []
    for l in range(min_len, max_len + 1):
        for prod in itertools.product('HT', repeat=l):
            seqs.append("".join(prod))
    return seqs

if __name__ == "__main__":
    seqs = generate_all_sequences(2, 8)
    # create 300 random pairs
    random.seed(42)
    candidates = []
    for _ in range(300):
        a = random.choice(seqs)
        b = random.choice(seqs)
        candidates.append({"sequence_a": a, "sequence_b": b})
    
    out_path = "/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/design/candidates.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Wrote {len(candidates)} candidates to {out_path}")
