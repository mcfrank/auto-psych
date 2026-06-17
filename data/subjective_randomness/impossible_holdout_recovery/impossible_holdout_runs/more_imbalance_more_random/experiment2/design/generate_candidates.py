import json
import itertools
import random
import os

all_seqs = []
for L in range(4, 9):
    for seq in itertools.product('HT', repeat=L):
        all_seqs.append("".join(seq))

candidates = []
# Ensure some same-length and some different-length pairs
# Let's sample 300 pairs randomly
random.seed(42)

# Same length pairs
same_len = []
for L in range(4, 9):
    seqs_L = [s for s in all_seqs if len(s) == L]
    pairs = list(itertools.combinations(seqs_L, 2))
    random.shuffle(pairs)
    same_len.extend(pairs[:50])

# Diff length pairs
diff_len = []
pairs = list(itertools.combinations(all_seqs, 2))
random.shuffle(pairs)
diff_len = [p for p in pairs if len(p[0]) != len(p[1])][:50]

selected = same_len + diff_len
# Also add some pairs that heavily contrast the features of interest:
# - high vs low imbalance
# - high vs low alternation rate

final_candidates = [{"sequence_a": a, "sequence_b": b} for a, b in selected]

os.makedirs("/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/design", exist_ok=True)
with open("/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/design/candidates.json", "w") as f:
    json.dump(final_candidates, f, indent=2)
print("Done writing", len(final_candidates), "candidates.")
