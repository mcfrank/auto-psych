# name: decade_vs_units_digit_share
# description: Among targets (not in the set) that share at least one digit with some example, yes rate when the target's tens digit equals some example's tens digit (same decade) minus yes rate when the shared digit(s) do not include a decade match; observed above null means people weight positional tens-digit/decade matches more than the model's position-free contains-digit concepts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    exi = np.where(valid, ex, -1000).astype(int)
    t = df["target"].astype(int).to_numpy()
    t2 = t[:, None]
    in_set = (exi == t2).any(axis=1)
    def digits(a):
        a = np.asarray(a)
        tens = np.where(a >= 10, (a // 10) % 10, -1)
        units = a % 10
        hund = np.where(a >= 100, 1, -1)
        return tens, units, hund
    et, eu, eh = digits(exi)
    tt, tu, th = digits(t2)
    tt0 = np.where(t2 >= 100, 0, tt)  # 100 has tens digit 0
    et0 = np.where(exi >= 100, 0, et)
    share = np.zeros(len(t), dtype=bool)
    for a in (et0, eu, eh):
        for b in (tt0, tu, th):
            share |= (valid & (a == b) & (a >= 0)).any(axis=1)
    decade = (valid & (et0 == tt0) & (tt0 >= 0)).any(axis=1)
    y = df["rating"].to_numpy().astype(float)
    base = share & ~in_set
    A = base & decade
    B = base & ~decade
    if A.sum() == 0 or B.sum() == 0:
        return 0.0
    return float(y[A].mean() - y[B].mean())
