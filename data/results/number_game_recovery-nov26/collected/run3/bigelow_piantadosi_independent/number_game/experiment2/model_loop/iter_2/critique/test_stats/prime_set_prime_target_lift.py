# name: prime_set_prime_target_lift
# description: Among sets of 2+ examples that are all prime, yes rate for non-example prime targets minus non-example odd composite targets; observed above null means people generalise along primality more sharply than the model, below means less.
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
    P = np.array([k > 1 and all(k % d for d in range(2, int(k ** 0.5) + 1)) for k in range(0, 101)])
    Ei = np.nan_to_num(E, nan=0).astype(int).clip(0, 100)
    allp = (P[Ei] | np.isnan(E)).all(1) & (n >= 2)
    tp = P[t.astype(int).clip(0, 100)]
    oddc = (t % 2 == 1) & ~tp & (t > 1)
    return _rate(r, allp & ~isex & tp) - _rate(r, allp & ~isex & oddc)
