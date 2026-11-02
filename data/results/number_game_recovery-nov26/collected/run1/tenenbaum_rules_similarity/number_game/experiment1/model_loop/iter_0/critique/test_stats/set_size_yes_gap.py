# name: set_size_yes_gap
# description: Yes rate on 1-2-example sets minus yes rate on 4-example sets (excluding targets that are examples); observed below the null mean means the model's size principle shrinks generalisation with more examples more strongly than humans do (and above means less strongly).
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    n = (~np.isnan(ex)).sum(axis=1)
    notin = ~(ex == t[:, None]).any(axis=1)
    a = notin & (n <= 2); b = notin & (n == 4)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
