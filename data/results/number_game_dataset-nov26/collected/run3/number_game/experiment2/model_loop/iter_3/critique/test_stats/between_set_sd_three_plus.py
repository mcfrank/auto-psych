# name: between_set_sd_three_plus
# description: For sets with 3+ examples, the standard deviation across sets of each set's mean non-example-target yes rate; observed above null means people's generalization differs more sharply between sets (some sets tight, some broad) than the model produces, below = model exaggerates between-set differences.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n >= 3]
    parts = sub["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    t = sub["target"].astype(int)
    is_ex = parts.eq(t, axis=0).any(axis=1)
    s = sub[~is_ex]
    means = s.groupby("number_set")["rating"].mean()
    if len(means) < 2:
        return 0.0
    return float(means.std())
