# name: power_of_two_set_specificity
# description: In sets of >=2 examples all in {2,4,8,16,32,64}, yes rate for non-example power-of-two targets minus yes rate for other even targets; t_observed above null_mean means people pick out the powers-of-two rule more sharply than the model's shared "powers" family prior allows, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    p2 = np.array([2, 4, 8, 16, 32, 64], dtype=float)
    ex_p2 = (np.isin(s, p2) | ~v).all(axis=1)
    is_ex = (np.where(v, s, -1) == t[:, None]).any(axis=1)
    tp2 = np.isin(t, p2.astype(int))
    r = df["rating"].to_numpy(dtype=float)
    base = (n >= 2) & ex_p2 & ~is_ex
    a = base & tp2; b = base & ~tp2 & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
