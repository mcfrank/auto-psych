# name: single_example_far_target_yes
# description: Yes rate for single-example sets on targets more than 30 away from the example; observed above null means the model generalizes too narrowly from one example to distant numbers, below means too broadly.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    n = (~np.isnan(ex)).sum(axis=1)
    m = (n == 1) & (np.abs(t - ex[:, 0]) > 30)
    return float(r[m].mean()) if m.any() else 0.0
