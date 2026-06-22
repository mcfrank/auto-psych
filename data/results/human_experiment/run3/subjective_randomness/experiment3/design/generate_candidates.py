import json
import random
import itertools


def generate_candidates():
    # Generate all possible sequences of lengths 2 to 8
    seqs = []
    for length in range(2, 9):
        for p in itertools.product("HT", repeat=length):
            seqs.append("".join(p))

    # Sample 300 random pairs
    random.seed(42)
    pairs = []

    # Let's ensure a mix of same-length and different-length pairs
    # by just sampling pairs randomly
    sampled_indices = set()
    while len(pairs) < 300:
        i, j = random.randint(0, len(seqs) - 1), random.randint(0, len(seqs) - 1)
        # Avoid exact identical sequences if we want, but it's okay, maybe they are less informative
        if i == j:
            continue
        if (i, j) not in sampled_indices:
            sampled_indices.add((i, j))
            pairs.append({"sequence_a": seqs[i], "sequence_b": seqs[j]})

    with open(
        "/scratch/users/benpry/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/design/candidates.json",
        "w",
    ) as f:
        json.dump(pairs, f, indent=2)


if __name__ == "__main__":
    generate_candidates()
