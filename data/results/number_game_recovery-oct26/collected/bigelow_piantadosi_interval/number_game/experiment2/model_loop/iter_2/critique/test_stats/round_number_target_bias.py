# name: round_number_target_bias
# description: Among sets not consisting only of multiples of 10, yes rate to non-example targets that are multiples of 10 minus yes rate to other even non-example targets; observed above the null means people over-endorse salient round numbers beyond what the model's concepts predict, below means the model over-endorses them.
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    all10 = ((parts % 10 == 0) | parts.isna()).all(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    is_ex = parts.eq(pd.Series(t, index=df.index), axis=0).any(axis=1).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    base = ~all10 & ~is_ex
    a = base & (t % 10 == 0)
    b = base & (t % 2 == 0) & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
