# name: power2_set_power2_target_lift
# description: On sets of 2+ examples that are all powers of 2, yes rate for non-example power-of-2 targets minus non-example even targets that are not powers of 2; observed above null means people commit to powers-of-2 more than the model (exponent cost too high), below means less.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    p2 = np.isin(np.arange(101), [1, 2, 4, 8, 16, 32, 64])
    Ei = np.nan_to_num(E, nan=0).astype(int).clip(0, 100)
    allp = (p2[Ei] | np.isnan(E)).all(1) & (n >= 2)
    tp = p2[t.astype(int).clip(0, 100)]
    return _rate(r, allp & ~isex & tp) - _rate(r, allp & ~isex & ~tp & (t % 2 == 0))
