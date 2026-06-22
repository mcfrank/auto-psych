import json
import itertools
import random
import os


def generate_all_sequences():
    seqs = []
    for length in range(2, 9):
        for p in itertools.product("HT", repeat=length):
            seqs.append("".join(p))
    return seqs


def main():
    out_dir = "/scratch/users/benpry/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/design"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "candidates.json")

    all_seqs = generate_all_sequences()

    # We want a diverse set of pairs.
    # Let's sample pairs randomly, but also try to get a mix of lengths.
    candidates = []

    # Sample 300 pairs randomly
    random.seed(42)

    # Let's do some stratified sampling based on length differences and lengths
    # There are 7 lengths (2 to 8).
    # 7 x 7 = 49 combinations of lengths.
    # We want 300 total, so about 6 per length combination.

    length_pairs = list(itertools.product(range(2, 9), repeat=2))
    seqs_by_length = {L: [s for s in all_seqs if len(s) == L] for L in range(2, 9)}

    for L1, L2 in length_pairs:
        # Sample 6 pairs for this length combination
        for _ in range(6):
            s1 = random.choice(seqs_by_length[L1])
            s2 = random.choice(seqs_by_length[L2])
            # Avoid identical sequences
            if s1 == s2 and len(seqs_by_length[L1]) > 1:
                while s1 == s2:
                    s2 = random.choice(seqs_by_length[L2])
            candidates.append({"sequence_a": s1, "sequence_b": s2})

    # That gives 49 * 6 = 294 pairs.

    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)


if __name__ == "__main__":
    main()
