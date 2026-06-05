import itertools
import random
import json

# 1. Generate all possible sequences for the allowed lengths
sequences = []
for length in allowed_sequence_lengths:
    for seq in itertools.product("HT", repeat=length):
        sequences.append("".join(seq))

# 2. Generate unique pairs (A < B to avoid symmetric duplicates and A=B)
all_pairs = []
for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
        all_pairs.append((sequences[i], sequences[j]))

# 3. Sample a subset of pairs to score (to keep runtime manageable)
# There are 56,280 unique pairs; we sample 3,000 to ensure a good mix of high EIG candidates.
random.seed(42)
num_to_sample = min(3000, len(all_pairs))
candidates = random.sample(all_pairs, num_to_sample)

# 4. Score each candidate using the provided EIG function
scored = []
for seq_a, seq_b in candidates:
    # expected_information_gain takes a single tuple argument
    eig = expected_information_gain((seq_a, seq_b))
    scored.append((eig, seq_a, seq_b))

# 5. Sort by EIG descending
scored.sort(key=lambda x: x[0], reverse=True)

# 6. Select top `total_trials` stimuli with a diversity constraint
# To ensure the experiment collectively discriminates across all theories,
# we prevent any single sequence from dominating the trials.
selected = []
seq_counts = {}
max_occurrences = 3

for eig, seq_a, seq_b in scored:
    if (
        seq_counts.get(seq_a, 0) < max_occurrences
        and seq_counts.get(seq_b, 0) < max_occurrences
    ):
        selected.append((eig, seq_a, seq_b))
        seq_counts[seq_a] = seq_counts.get(seq_a, 0) + 1
        seq_counts[seq_b] = seq_counts.get(seq_b, 0) + 1
    if len(selected) == total_trials:
        break

# Fallback if the diversity constraint was too strict to find enough trials
if len(selected) < total_trials:
    for eig, seq_a, seq_b in scored:
        if (eig, seq_a, seq_b) not in selected:
            selected.append((eig, seq_a, seq_b))
        if len(selected) == total_trials:
            break

# 7. Format and write outputs
stimuli_list = [
    {"sequence_a": seq_a, "sequence_b": seq_b, "eig": float(eig)}
    for eig, seq_a, seq_b in selected
]

(out_dir / "stimuli.json").write_text(json.dumps(stimuli_list, indent=2))

min_eig = min(s["eig"] for s in stimuli_list)
max_eig = max(s["eig"] for s in stimuli_list)

rationale = f"""# Design Rationale

- **Total Trials**: {total_trials}
- **EIG Range**: {min_eig:.4f} to {max_eig:.4f}
- **Candidate Generation**: Generated all possible sequences of lengths {allowed_sequence_lengths}. Sampled {num_to_sample} unique pairs to evaluate.
- **Selection Strategy**: Scored pairs using the provided `expected_information_gain` function. Selected the top {total_trials} pairs with the highest EIG. Applied a diversity constraint (maximum {max_occurrences} occurrences per sequence) to ensure the final design covers a broad range of sequence features (alternations, proportions, lengths). This diverse set effectively discriminates between the Rational Representativeness, Alternation Heuristic, and Proportion Heuristic models.
"""

(out_dir / "design_rationale.md").write_text(rationale.strip())
