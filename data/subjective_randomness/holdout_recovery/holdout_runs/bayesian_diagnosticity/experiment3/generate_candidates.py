import json
import random
import itertools

def generate_sequences(min_len=4, max_len=8):
    seqs = []
    for l in range(min_len, max_len + 1):
        for prod in itertools.product('HT', repeat=l):
            seqs.append(''.join(prod))
    return seqs

def main():
    seqs = generate_sequences(4, 8)
    candidates = []
    
    random.seed(42)
    pairs = set()
    pairs.add(("HHHHH", "THTHT"))
    pairs.add(("HHHHHHHH", "THTHTHTH"))
    pairs.add(("HTHTHT", "HHHTTT"))
    pairs.add(("HHTT", "HTHT"))
    
    while len(pairs) < 100:
        a = random.choice(seqs)
        b = random.choice(seqs)
        if a != b:
            if a < b:
                pairs.add((a, b))
            else:
                pairs.add((b, a))
                
    for a, b in pairs:
        candidates.append({"sequence_a": a, "sequence_b": b})
        
    out_dir = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/bayesian_diagnosticity/experiment3/design"
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
        
if __name__ == "__main__":
    main()
