# name: contains_digit_decade_yes
# description: Sets of >=2 examples that all contain a digit d but do not all end in d: yes-rate on non-example targets whose tens digit is d and that are >3 from every example; observed above null_mean means people generalise a "contains digit d" rule into the d-decade more than the model's discounted digit rules allow, below means the model over-produces it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    dig = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        ds = [d for d in range(1, 10) if len(xs) >= 2 and all(str(d) in str(x) for x in xs) and not all(x % 10 == d for x in xs)]
        dig[s] = ds[0] if ds else -1
    d = sets.map(dig).values
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (d >= 0) & (t // 10 == d) & (t < 100) & (dist > 3)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
