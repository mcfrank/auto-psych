# name: perfect_alternation_choice_rate
# description: Proportion of choices for a perfectly alternating sequence (HTHT.../THTH..., length>=4) when it is paired with a non-alternating one; observed above null_mean means the model over-penalises perfect alternation (people find it more random than predicted), below means it under-penalises it.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    seqs = pd.unique(np.concatenate([a, b]))
    alt = {s: (len(s) >= 4 and all(s[i] != s[i - 1] for i in range(1, len(s)))) for s in seqs}
    pa = df["sequence_a"].map(alt).values.astype(bool)
    pb = df["sequence_b"].map(alt).values.astype(bool)
    mask = pa ^ pb
    if mask.sum() == 0:
        return 0.0
    y = df["chose_left"].values[mask]
    chose_alt = np.where(pa[mask], y, 1 - y)
    return float(chose_alt.mean())
