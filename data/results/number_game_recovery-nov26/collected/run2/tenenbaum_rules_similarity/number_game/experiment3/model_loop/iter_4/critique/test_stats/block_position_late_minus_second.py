# name: block_position_late_minus_second
# description: Within runs of consecutive trials with the same set, yes rate on non-example targets at run position >= 3 (third query onward) minus at position 2 (second query); observed above null means willingness to say "yes" keeps growing with repeated queries about a set, beyond the model's single first-trial shift (below null = it decays).
def test_statistic(df):
    ns = df["number_set"].astype(str).values
    ti = df["trial_index"].values
    prev = np.concatenate([[""], ns[:-1]])
    first = (ns != prev) | (ti == 0)
    blk = np.cumsum(first)
    pos = pd.Series(np.zeros(len(df))).groupby(blk).cumcount().values
    tgt = df["target"].astype(int).astype(str).values
    is_ex = np.array([t in s.split(",") for s, t in zip(ns, tgt)])
    r = df["rating"].values.astype(float)
    a = (pos >= 2) & ~is_ex
    b = (pos == 1) & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
