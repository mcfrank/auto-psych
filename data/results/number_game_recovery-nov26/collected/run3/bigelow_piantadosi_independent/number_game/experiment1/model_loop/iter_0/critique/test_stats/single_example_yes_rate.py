# name: single_example_yes_rate
# description: Yes-rate on trials whose set has exactly one example, minus the yes-rate on trials with 3-4 examples; observed > null means people generalize from one example relatively more broadly than the model's size principle predicts, observed < null means more narrowly.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    return ex, t

def test_statistic(df):
    ex, t = _parse(df)
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    a = n == 1; b = n >= 3
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
