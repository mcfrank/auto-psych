# name: round_target_bias
# description: Among trials whose set contains no multiple of 10, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed > model means people have a prior preference for round numbers that example-based similarity (digit/parity/factor match to the examples) cannot produce, observed < model means the model over-generalizes to round targets.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").values.astype(float)
    has10 = np.any((E % 10) == 0, axis=1)
    m = ~has10
    a = m & (t % 10 == 0)
    b = m & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
