# name: balanced_count_equal_span
# description: Among pairs with equal tally span (max-min of running H-T) but different final |#H-#T|, the rate of choosing the sequence with the more balanced final H/T count; observed above null means people use the overall H/T count beyond the span the model uses (model under-produces count balancing), below means the opposite.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def span(s):
        t = np.cumsum([1 if ch == "H" else -1 for ch in s])
        return int(max(t.max(), 0) - min(t.min(), 0))
    sp = {s: span(s) for s in seqs}
    ia = (2 * df["sequence_a"].str.count("H") - df["sequence_a"].str.len()).abs().to_numpy()
    ib = (2 * df["sequence_b"].str.count("H") - df["sequence_b"].str.len()).abs().to_numpy()
    pa = df["sequence_a"].map(sp).to_numpy(); pb = df["sequence_b"].map(sp).to_numpy()
    m = (pa == pb) & (ia != ib)
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(ia[m] < ib[m], c, 1 - c).mean())
