# name: four_minus_three_example_nonexample_rate
# description: Yes rate for non-example targets on 4-example sets minus on 3-example sets; observed above null means the model generalizes too narrowly from 4 examples relative to 3 (people broaden again with 4 examples, unlike the size principle).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    a = (~is_ex) & (n == 4)
    b = (~is_ex) & (n == 3)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
