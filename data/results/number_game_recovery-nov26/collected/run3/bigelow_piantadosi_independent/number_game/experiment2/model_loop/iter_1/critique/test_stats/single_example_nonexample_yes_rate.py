# name: single_example_nonexample_yes_rate
# description: Yes rate for non-example targets after a single-example set; observed above null means the model generalizes too narrowly from one example (too strong size principle/commitment), below means too broadly.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    n = (~np.isnan(E)).sum(1)
    m = (n == 1) & ~(E == t[:, None]).any(1)
    return _rate(r, m)
