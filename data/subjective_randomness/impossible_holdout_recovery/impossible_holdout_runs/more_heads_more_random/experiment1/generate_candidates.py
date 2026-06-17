import json
import random
import itertools

def generate_sequences():
    seqs = []
    for length in range(3, 9):
        for p in itertools.product('HT', repeat=length):
            seqs.append("".join(p))
    return seqs

def main():
    random.seed(42)
    seqs = generate_sequences()
    
    candidates = []
    # Let's just pick 300 random pairs
    for _ in range(300):
        a = random.choice(seqs)
        b = random.choice(seqs)
        while a == b:
            b = random.choice(seqs)
        candidates.append({"sequence_a": a, "sequence_b": b})
    
    out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_heads_more_random/experiment1/design/candidates.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Wrote {len(candidates)} candidates to {out_path}")

if __name__ == "__main__":
    main()
