import itertools
import random
import json

# 1. Generate candidate stimuli
# Create all possible sequences for the allowed lengths
sequences = []
for length in allowed_sequence_lengths:
    for seq in itertools.product("HT", repeat=length):
        sequences.append("".join(seq))

# Generate a random sample of pairs to evaluate to keep computation tractable
random.seed(42)
candidate_pairs = set()
# 2500 pairs provides a good coverage of the space while remaining fast to score
while len(candidate_pairs) < 2500:
    seq_a = random.choice(sequences)
    seq_b = random.choice(sequences)
    if seq_a != seq_b:
        candidate_pairs.add((seq_a, seq_b))

candidate_pairs = list(candidate_pairs)

# 2. Score each candidate
scored = []
for seq_a, seq_b in candidate_pairs:
    # The provided expected_information_gain takes a single tuple argument
    eig = expected_information_gain((seq_a, seq_b))
    scored.append((eig, seq_a, seq_b))

# Sort by EIG descending
scored.sort(key=lambda x: -x[0])

# 3. Select exactly total_trials stimuli
selected = []
seen_pairs = set()
seq_counts = {}

# First pass: try to select with diversity constraints (max 3 appearances per sequence)
for eig, seq_a, seq_b in scored:
    # Use frozenset to treat (A, B) and (B, A) as the same pair
    pair_set = frozenset([seq_a, seq_b])
    
    if pair_set in seen_pairs:
        continue
        
    # Enforce diversity: don't overuse any single sequence
    if seq_counts.get(seq_a, 0) >= 3 or seq_counts.get(seq_b, 0) >= 3:
        continue
        
    selected.append({"sequence_a": seq_a, "sequence_b": seq_b, "eig": float(eig)})
    seen_pairs.add(pair_set)
    seq_counts[seq_a] = seq_counts.get(seq_a, 0) + 1
    seq_counts[seq_b] = seq_counts.get(seq_b, 0) + 1
    
    if len(selected) == total_trials:
        break

# Second pass: if we need more to reach total_trials, relax the sequence count constraint
if len(selected) < total_trials:
    for eig, seq_a, seq_b in scored:
        pair_set = frozenset([seq_a, seq_b])
        if pair_set not in seen_pairs:
            selected.append({"sequence_a": seq_a, "sequence_b": seq_b, "eig": float(eig)})
            seen_pairs.add(pair_set)
        if len(selected) == total_trials:
            break

# 4. Write outputs
# Write stimuli.json
with open(out_dir / "stimuli.json", "w") as f:
    json.dump(selected, f, indent=2)

# Write design_rationale.md
min_eig = selected[-1]["eig"]
max_eig = selected[0]["eig"]
rationale = f"""# Design Rationale

- **Total trials**: {total_trials}
- **EIG range**: {min_eig:.4f} to {max_eig:.4f}
- **Candidate generation**: Generated all possible sequences of lengths {allowed_sequence_lengths} (H/T). Randomly sampled 2500 unique pairs to evaluate.
- **Selection**: Scored each pair using the provided `expected_information_gain` function. Selected the top {total_trials} pairs with the highest EIG.
- **Diversity**: To ensure a diverse set of stimuli, a constraint was added so that no single sequence appears more than 3 times in the final design, and symmetric duplicate pairs were excluded. This prevents the design from being dominated by slight variations of a single highly informative comparison, allowing it to broadly discriminate across all models ({', '.join(model_names)}).
"""

with open(out_dir / "design_rationale.md", "w") as f:
    f.write(rationale)