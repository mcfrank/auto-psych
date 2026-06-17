import json
import random
import itertools

def generate_all_sequences(min_length=4, max_length=8):
    seqs = []
    for length in range(min_length, max_length + 1):
        for p in itertools.product('HT', repeat=length):
            seqs.append(''.join(p))
    return seqs

def main():
    random.seed(42)
    all_seqs = generate_all_sequences()
    candidates = []
    
    # Let's generate about 300 random pairs
    # To have a good variety of lengths and imbalances:
    for _ in range(300):
        seq_a = random.choice(all_seqs)
        seq_b = random.choice(all_seqs)
        candidates.append({"sequence_a": seq_a, "sequence_b": seq_b})
        
    out_path = "/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment3/design/candidates.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
        
if __name__ == "__main__":
    main()
