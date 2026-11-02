# name: tens_vs_units_digit_match
# description: Among non-example targets 10-99 in sets with any two-digit example, yes rate for targets whose tens digit equals some example's tens digit (and units matches none) minus yes rate for targets whose units digit equals some example's units digit (and tens matches none); observed above the null means the model under-weights decade (tens-position) similarity relative to last-digit similarity, below means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    two = (ex >= 10) & (ex <= 99)
    et = np.where(two, np.floor(np.nan_to_num(ex) / 10), -1)
    eu = np.where(two, np.nan_to_num(ex) % 10, -1)
    tt = (t // 10)[:, None]; tu = (t % 10)[:, None]
    tm = np.any(et == tt, axis=1); um = np.any(eu == tu, axis=1)
    inset = np.any(ex == t[:, None], axis=1)
    base = (t >= 10) & (t <= 99) & ~inset & two.any(axis=1)
    a = y[base & tm & ~um]; b = y[base & um & ~tm]
    return float((a.mean() if a.size else 0.0) - (b.mean() if b.size else 0.0))
