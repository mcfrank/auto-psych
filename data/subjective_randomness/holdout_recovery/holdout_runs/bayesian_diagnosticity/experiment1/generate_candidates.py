import json
import itertools
import random
import os

def generate_sequences(max_len=8):
    seqs = []
    for length in range(1, max_len + 1):
        for prod in itertools.product('HT', repeat=length):
            seqs.append("".join(prod))
    return seqs

def main():
    random.seed(42)
    seqs = generate_sequences(8)
    
    candidates = []
    
    # Randomly sample some pairs
    # EIG script will evaluate all candidates, so we don't want too many (maybe 500-1000)
    # Let's create a curated list of pairs:
    # 1. same length pairs
    # 2. different length pairs
    # We'll randomly select 1000 pairs
    
    num_candidates = 500
    
    for _ in range(num_candidates):
        a = random.choice(seqs)
        b = random.choice(seqs)
        # Avoid identical sequences
        while a == b:
            b = random.choice(seqs)
            
        candidates.append({
            "sequence_a": a,
            "sequence_b": b
        })
        
    design_dir = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/bayesian_diagnosticity/experiment1/design"
    os.makedirs(design_dir, exist_ok=True)
    out_path = os.path.join(design_dir, "candidates.json")
    
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
        
    print(f"Wrote {len(candidates)} candidates to {out_path}")

if __name__ == "__main__":
    main()
