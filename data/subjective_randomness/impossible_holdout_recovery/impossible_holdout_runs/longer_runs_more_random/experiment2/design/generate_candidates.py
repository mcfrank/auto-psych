import json
import random
import itertools

def generate_all_sequences(min_len, max_len):
    seqs = []
    for length in range(min_len, max_len + 1):
        for p in itertools.product("HT", repeat=length):
            seqs.append("".join(p))
    return seqs

def main():
    random.seed(42)
    seqs = generate_all_sequences(3, 8)
    
    candidates = []
    
    # Try to make candidate pairs somewhat diverse.
    # 300 pairs total.
    for _ in range(300):
        seq_a = random.choice(seqs)
        seq_b = random.choice(seqs)
        # avoid identical
        while seq_a == seq_b:
            seq_b = random.choice(seqs)
        
        candidates.append({
            "sequence_a": seq_a,
            "sequence_b": seq_b
        })
        
    out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/longer_runs_more_random/experiment2/design/candidates.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)

if __name__ == "__main__":
    main()
