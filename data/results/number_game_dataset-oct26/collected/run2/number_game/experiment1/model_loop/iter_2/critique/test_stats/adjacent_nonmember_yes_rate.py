# name: adjacent_nonmember_yes_rate
# description: Among multi-example sets, yes rate for targets not in the set that lie within 3 of some example but share no final digit with any example; observed < model means the model over-generalizes by numeric closeness to near neighbours.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values[:, None]
    y = df["rating"].astype(float).values
    valid = ~np.isnan(ex)
    multi = valid.sum(1) >= 2
    d = np.where(valid, np.abs(t - ex), np.inf)
    inset = (d == 0).any(1)
    near = (d <= 3).any(1)
    digit = (valid & (np.nan_to_num(ex) % 10 == t % 10)).any(1)
    m = multi & ~inset & near & ~digit
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
