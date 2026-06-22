import json
import random
import itertools


def generate_candidates():
    # Generate all sequences of length 2 to 8
    seqs = []
    for length in range(2, 9):
        for p in itertools.product("HT", repeat=length):
            seqs.append("".join(p))

    # Sample 300 random pairs
    pairs = []
    # We want a diverse set of pairs, random sampling is usually fine for candidate generation
    # if the pool is 300 out of 129k, we will get a good spread.

    # Actually let's make sure we get a variety of length combinations
    all_pairs = list(itertools.combinations(seqs, 2))
    sampled = random.sample(all_pairs, 300)

    candidates = [{"sequence_a": p[0], "sequence_b": p[1]} for p in sampled]

    with open(
        "/scratch/users/benpry/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/design/candidates.json",
        "w",
    ) as f:
        json.dump(candidates, f, indent=2)


if __name__ == "__main__":
    generate_candidates()
