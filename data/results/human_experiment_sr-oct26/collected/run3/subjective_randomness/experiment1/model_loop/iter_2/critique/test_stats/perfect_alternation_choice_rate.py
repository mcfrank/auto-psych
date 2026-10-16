# name: perfect_alternation_choice_rate
# description: Proportion of choices for a perfectly alternating sequence (HTHT.../THTH..., length>=4) when paired with a non-alternating one; observed above null_mean means the model over-penalises perfect alternation (people find it more random than predicted), below means it under-penalises it.
def test_statistic(df):
    def alt(s):
        return len(s) >= 4 and all(s[i] != s[i + 1] for i in range(len(s) - 1))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    pa = df["sequence_a"].map(m).astype(bool)
    pb = df["sequence_b"].map(m).astype(bool)
    sel = pa ^ pb
    if sel.sum() == 0:
        return 0.5
    c = np.where(pa[sel], df["chose_left"][sel], 1 - df["chose_left"][sel])
    return float(np.mean(c))
