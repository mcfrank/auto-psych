import json
import random
import itertools

def generate_all_sequences(min_len, max_len):
    seqs = []
    for length in range(min_len, max_len + 1):
        for prod in itertools.product("HT", repeat=length):
            seqs.append("".join(prod))
    return seqs

def main():
    # max sequence length is 8
    # let's generate sequences of length 4 to 8
    seqs = generate_all_sequences(4, 8)
    
    # We want roughly 300 candidates. Let's sample pairs.
    # To ensure good coverage, we can sample pairs randomly but make sure they are distinct.
    random.seed(42)
    
    candidates = []
    seen = set()
    
    # Let's add some structured pairs
    # same length
    # different lengths
    
    while len(candidates) < 300:
        a = random.choice(seqs)
        b = random.choice(seqs)
        if a == b:
            continue
        
        # normalize pair so (A,B) and (B,A) are considered the same
        pair = tuple(sorted([a, b]))
        if pair not in seen:
            seen.add(pair)
            candidates.append({"sequence_a": a, "sequence_b": b})

    with open("candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    
    print(f"Generated {len(candidates)} candidates.")

if __name__ == "__main__":
    main()
