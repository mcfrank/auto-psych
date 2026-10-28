# name: even_set_parity_contrast
# description: For sets of 2-4 examples that are all even, yes rate on even non-example targets minus odd targets; observed above null_mean means people use the broad "even numbers" rule more than the model predicts, below means the model over-weights parity for even sets.
def test_statistic(df):
    u = df["number_set"].unique()
    ok = {s: (lambda e: len(e) >= 2 and all(v % 2 == 0 for v in e))([int(x) for x in str(s).split(",")]) for s in u}
    sel = df["number_set"].map(ok).astype(bool)
    t = df["target"].astype(int)
    isex = pd.Series([str(tt) in str(s).split(",") for s, tt in zip(df["number_set"], t)], index=df.index)
    r = df["rating"].astype(float)
    ev = sel & ~isex & (t % 2 == 0)
    od = sel & (t % 2 == 1)
    if ev.sum() == 0 or od.sum() == 0:
        return 0.0
    return float(r[ev].mean() - r[od].mean())
