import json
import random
import itertools

def generate_sequences():
    seqs = []
    for length in range(3, 9):
        for p in itertools.product('HT', repeat=length):
            seqs.append("".join(p))
    return seqs

all_seqs = generate_sequences()
candidates = []

random.seed(101) # new seed
pairs = set()
while len(pairs) < 60:
    a, b = random.sample(all_seqs, 2)
    pairs.add((a, b))

for a, b in pairs:
    candidates.append({"sequence_a": a, "sequence_b": b})

with open("/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/bayesian_diagnosticity/experiment2/design/candidates.json", "w") as f:
    json.dump(candidates, f)
