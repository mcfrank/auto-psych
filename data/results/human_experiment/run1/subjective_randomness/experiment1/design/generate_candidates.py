import json
import random
import itertools


def generate_all_sequences(min_len=2, max_len=8):
    seqs = []
    for length in range(min_len, max_len + 1):
        for p in itertools.product("HT", repeat=length):
            seqs.append("".join(p))
    return seqs


if __name__ == "__main__":
    random.seed(42)
    seqs = generate_all_sequences()
    candidates = []

    # We want ~300 diverse candidates.
    # Let's sample 300 random pairs.
    # To avoid duplicates, we can sample without replacement.
    all_pairs_indices = list(itertools.combinations(range(len(seqs)), 2))
    sampled_indices = random.sample(all_pairs_indices, 300)

    for i, j in sampled_indices:
        if random.random() > 0.5:
            candidates.append({"sequence_a": seqs[i], "sequence_b": seqs[j]})
        else:
            candidates.append({"sequence_a": seqs[j], "sequence_b": seqs[i]})

    with open(
        "/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/design/candidates.json",
        "w",
    ) as f:
        json.dump(candidates, f, indent=2)
    print(f"Generated {len(candidates)} candidates.")
