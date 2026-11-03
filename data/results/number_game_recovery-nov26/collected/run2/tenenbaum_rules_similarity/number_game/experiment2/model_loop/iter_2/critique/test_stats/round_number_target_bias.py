# name: round_number_target_bias
# description: Among sets not consisting entirely of multiples of 5, yes rate for non-example targets that are multiples of 10 minus yes rate for other non-example targets; observed above the null means people favour round-number targets beyond what the model's rules predict (below = they disfavour them).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    m5 = {s: all(int(x) % 5 == 0 for x in s.split(",")) for s in sets.unique()}
    keep = ~sets.map(m5).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    exm = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    inset = (exm == df["target"].astype(int).to_numpy()[:, None]).any(axis=1)
    base = keep & ~inset
    a = base & (t % 10 == 0)
    b = base & (t % 10 != 0)
    r = df["rating"].to_numpy()
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
