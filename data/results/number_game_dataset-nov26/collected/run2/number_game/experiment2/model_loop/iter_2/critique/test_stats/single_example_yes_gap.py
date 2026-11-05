# name: single_example_yes_gap
# description: Yes-rate on single-example sets minus yes-rate on multi-example sets (non-example targets only); observed above null means the model under-generalizes from one example (too narrow/conservative on n=1), below means it over-generalizes.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    a = (n == 1) & (~inset)
    b = (n > 1) & (~inset)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
