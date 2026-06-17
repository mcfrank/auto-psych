import json
import random
import itertools

def get_all_seqs(max_len):
    seqs = []
    for l in range(1, max_len + 1):
        for p in itertools.product('HT', repeat=l):
            seqs.append("".join(p))
    return seqs

def generate_candidates():
    seqs = get_all_seqs(8)
    candidates = []
    random.seed(42)
    # 500 totally random pairs
    for _ in range(500):
        s1 = random.choice(seqs)
        s2 = random.choice(seqs)
        if s1 != s2:
            candidates.append({"sequence_a": s1, "sequence_b": s2})
    
    # 200 pairs of the exact same length
    for _ in range(200):
        l = random.randint(3, 8)
        s1 = "".join(random.choices('HT', k=l))
        s2 = "".join(random.choices('HT', k=l))
        if s1 != s2:
            candidates.append({"sequence_a": s1, "sequence_b": s2})
            
    # Some extreme contrasts
    extremes = ["HTHTHTHT", "THTHTHTH", "HHHHHHHH", "TTTTTTTT", "HHTTHHTT", "HHHHTTTT"]
    for s1 in extremes:
        for s2 in extremes:
            if s1 != s2:
                candidates.append({"sequence_a": s1, "sequence_b": s2})
                
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        pair = tuple(sorted([c["sequence_a"], c["sequence_b"]]))
        if pair not in seen:
            seen.add(pair)
            unique_candidates.append(c)
            
    # Subsample to exactly 500 to keep it fast
    if len(unique_candidates) > 500:
        unique_candidates = random.sample(unique_candidates, 500)
            
    with open('/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/encoding_compressibility/experiment2/design/candidates.json', 'w') as f:
        json.dump(unique_candidates, f, indent=2)

if __name__ == "__main__":
    generate_candidates()